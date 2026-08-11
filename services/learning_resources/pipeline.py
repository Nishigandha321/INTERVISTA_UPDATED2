# services/learning_resources/pipeline.py
# Main orchestration for resource retrieval pipeline

import asyncio
import dataclasses
from typing import List, Dict, Any, Tuple, Optional
from .normalizer import LearningResource, ResourceNormalizer
from .query_generator import SearchQueryGenerator
from .ranker import SemanticRanker
from .fetchers import YouTubeFetcher, GeeksForGeeksFetcher, BaseFetcher
from core.llm.llm_service import LLMService
from core.prompts.prompt_manager import PromptManager
from utils.logger import Logger

logger = Logger(__name__)

# Fields belonging to LearningResource — used when reconstructing from cached metadata
_LEARNING_RESOURCE_FIELDS = {f.name for f in dataclasses.fields(LearningResource)}


class ResourceRetrievalPipeline:
    """
    Context-aware, cache-first learning resource retrieval pipeline.

    Flow for each weak concept (up to 3 concepts × top_k=3 resources):
    1. Build a context-enriched search string and query the FAISS cache.
    2. Reuse partial cache hits when available; fetch only enough to fill top_k.
    3. On cache miss: generate LLM search queries → parallel fetch (YouTube + GFG)
       → normalise → deduplicate → semantic rank → store embeddings → return.
    """

    def __init__(
        self,
        llm_service: LLMService,
        prompt_manager: PromptManager,
        youtube_api_key: str = None,
        ranker: Optional[SemanticRanker] = None,
        vector_store=None,
        cache_similarity_threshold: float = 0.50,
        min_cache_results: Optional[int] = None,
    ):
        self.llm_service = llm_service
        self.prompt_manager = prompt_manager
        self.ranker = ranker
        self.vector_store = vector_store
        self.cache_similarity_threshold = cache_similarity_threshold
        self._default_min_cache_results = min_cache_results
        self._http_semaphore = asyncio.Semaphore(5)

        self.query_generator = SearchQueryGenerator(llm_service, prompt_manager)
        self.fetchers: List[BaseFetcher] = [
            YouTubeFetcher(api_key=youtube_api_key),
            GeeksForGeeksFetcher(use_api=False),
        ]

        logger.info(
            f"ResourceRetrievalPipeline initialized "
            f"(ranker: {'enabled' if ranker else 'disabled'}, "
            f"vector_cache: {'enabled' if vector_store else 'disabled'}, "
            f"cache_threshold: {cache_similarity_threshold})"
        )

    # ==================================================================
    # PUBLIC ENTRY POINT
    # ==================================================================

    async def retrieve_and_rank(
        self,
        weak_concepts: List[str],
        role: str = "Software Engineer",
        top_k: int = 3,
        queries_per_concept: int = 4,
        results_per_query: int = 3,
        min_score: float = 0.0,
        level: str = "",
        company: str = "",
        difficulty: str = "",
        cache_similarity_threshold: Optional[float] = None,
        min_cache_results: Optional[int] = None,
    ) -> Dict[str, List[Tuple[LearningResource, float]]]:
        """
        Context-aware, cache-first retrieval for a list of weak concepts.

        Concepts are retrieved in parallel; failures are isolated per concept.
        """
        if not weak_concepts:
            logger.warning("No concepts provided")
            return {}

        threshold = (
            cache_similarity_threshold
            if cache_similarity_threshold is not None
            else self.cache_similarity_threshold
        )
        min_hits = (
            min_cache_results
            if min_cache_results is not None
            else (self._default_min_cache_results or top_k)
        )

        logger.info(
            f"retrieve_and_rank: {len(weak_concepts)} concepts, top_k={top_k}, "
            f"role={role}, level={level or 'n/a'}, company={company or 'n/a'}, "
            f"difficulty={difficulty or 'n/a'}, cache_threshold={threshold:.2f}"
        )

        async def _retrieve_one(concept: str) -> Tuple[str, List[Tuple[LearningResource, float]]]:
            try:
                ranked = await self._retrieve_concept(
                    concept=concept,
                    role=role,
                    top_k=top_k,
                    queries_per_concept=queries_per_concept,
                    results_per_query=results_per_query,
                    min_score=min_score,
                    level=level,
                    company=company,
                    difficulty=difficulty,
                    threshold=threshold,
                    min_hits=min_hits,
                )
                return concept, ranked
            except Exception as e:
                logger.error(f"Failed to retrieve resources for '{concept}': {e}")
                return concept, []

        pairs = await asyncio.gather(*[_retrieve_one(c) for c in weak_concepts])
        ranked_result = {concept: ranked for concept, ranked in pairs}

        total = sum(len(v) for v in ranked_result.values())
        logger.info(
            f"retrieve_and_rank complete: {total} total resources "
            f"across {len(ranked_result)} concepts"
        )
        return ranked_result

    # ==================================================================
    # PER-CONCEPT ORCHESTRATION
    # ==================================================================

    async def _retrieve_concept(
        self,
        concept: str,
        role: str,
        top_k: int,
        queries_per_concept: int,
        results_per_query: int,
        min_score: float,
        level: str,
        company: str,
        difficulty: str,
        threshold: float,
        min_hits: int,
    ) -> List[Tuple[LearningResource, float]]:
        """Cache-first retrieval for a single concept with partial cache reuse."""

        cache_query = self._build_cache_query(concept, role, level, company, difficulty)

        cached = self._search_cache(cache_query, top_k=top_k * 2, threshold=threshold)
        cached_list: List[Tuple[LearningResource, float]] = list(cached or [])

        if len(cached_list) >= min_hits:
            logger.info(f"Cache HIT for '{concept}': {len(cached_list)} resources")
            if min_score > 0:
                cached_list = [(r, s) for r, s in cached_list if s >= min_score]
            return cached_list[:top_k]

        slots_needed = top_k - len(cached_list)
        cache_count = len(cached_list)
        logger.info(
            f"Cache PARTIAL/MISS for '{concept}': {cache_count} hits (need {min_hits}) "
            f"— fetching {slots_needed} externally"
        )

        if slots_needed <= 0:
            return cached_list[:top_k]

        queries = await self.query_generator.generate_queries(
            concept,
            role=role,
            count=queries_per_concept,
            level=level,
            company=company,
            difficulty=difficulty,
        )
        if not queries:
            logger.warning(f"No queries generated for concept: '{concept}'")
            return cached_list[:top_k]

        raw_by_source = await self._fetch_for_queries(queries, results_per_query)
        normalized = await self._normalize_resources(raw_by_source)
        deduped = self._deduplicate_by_url(normalized)

        cached_urls = {r.url.lower().strip() for r, _ in cached_list}
        deduped = [r for r in deduped if r.url.lower().strip() not in cached_urls]

        if not deduped:
            logger.warning(f"No new external resources for '{concept}'")
            return cached_list[:top_k]

        if self.ranker:
            ranked_new = self.ranker.rank_resources(
                weak_concept=concept, resources=deduped, role=role
            )
            if min_score > 0:
                ranked_new = self.ranker.filter_by_threshold(ranked_new, min_score)
            ranked_new = ranked_new[:slots_needed]
        else:
            ranked_new = [(r, 0.5) for r in deduped[:slots_needed]]

        if self.vector_store and ranked_new:
            self._cache_resources_in_vector_store(ranked_new)

        combined = cached_list + ranked_new
        combined.sort(key=lambda t: t[1], reverse=True)
        combined = combined[:top_k]

        logger.info(
            f"External fetch for '{concept}': {len(ranked_new)} new + {cache_count} cached"
            + (f" (top score={combined[0][1]:.2f})" if combined else "")
        )
        return combined

    # ==================================================================
    # CACHE SEARCH
    # ==================================================================

    @staticmethod
    def _build_cache_query(
        concept: str,
        role: str,
        level: str,
        company: str,
        difficulty: str,
    ) -> str:
        parts = [concept]
        if role:
            parts.append(role)
        if level:
            parts.append(level)
        if difficulty:
            parts.append(difficulty)
        if company:
            parts.append(company)
        return " | ".join(parts)

    def _search_cache(
        self,
        cache_query: str,
        top_k: int,
        threshold: float,
    ) -> Optional[List[Tuple[LearningResource, float]]]:
        if not self.vector_store:
            return None

        try:
            raw_hits = self.vector_store.search_resources(cache_query, k=top_k)
        except Exception as e:
            logger.warning(f"Vector store search failed: {e}")
            return None

        if not raw_hits:
            return []

        results: List[Tuple[LearningResource, float]] = []
        for resource_id, raw_similarity, metadata in raw_hits:
            score = raw_similarity / 100.0
            if score < threshold:
                continue
            resource = self._resource_from_metadata(resource_id, metadata)
            if resource is None:
                continue
            stored_score = metadata.get("_rank_score")
            blended = (
                0.6 * float(stored_score) + 0.4 * score
                if stored_score is not None
                else score
            )
            results.append((resource, blended))

        results.sort(key=lambda t: t[1], reverse=True)
        return results

    @staticmethod
    def _resource_from_metadata(
        resource_id: str, metadata: Dict[str, Any]
    ) -> Optional[LearningResource]:
        if not metadata:
            return None
        try:
            clean = {k: v for k, v in metadata.items() if k in _LEARNING_RESOURCE_FIELDS}
            if not clean.get("url") or not clean.get("title"):
                return None
            return LearningResource(**clean)
        except Exception as e:
            logger.warning(f"Could not reconstruct LearningResource for {resource_id}: {e}")
            return None

    # ==================================================================
    # CACHE WRITE
    # ==================================================================

    def _cache_resources_in_vector_store(
        self,
        ranked_resources: List[Tuple[LearningResource, float]],
    ) -> None:
        if not self.vector_store:
            return
        batch = []
        for resource, score in ranked_resources:
            metadata = resource.to_dict()
            metadata["_rank_score"] = score
            batch.append((resource.id, resource.to_embedding_text(), metadata))
        try:
            self.vector_store.add_resources_batch(batch)
            logger.debug(f"Cached {len(batch)} resources in vector store")
        except Exception as e:
            logger.warning(f"Failed to cache resources in vector store: {e}")

    # ==================================================================
    # LLM RE-RANKING & EXPLANATIONS
    # ==================================================================

    async def generate_explanations(
        self,
        ranked_results: Dict[str, List[Tuple["LearningResource", float]]],
        role: str = "",
        level: str = "",
        difficulty: str = "",
    ) -> Dict[str, Dict[str, str]]:
        if not self.llm_service:
            return {}

        async def _explain_one(
            concept: str,
            resource_list: List[Tuple["LearningResource", float]],
        ) -> Tuple[str, Dict[str, str]]:
            if not resource_list:
                return concept, {}
            try:
                explanations = await self._explain_concept_resources(
                    concept=concept,
                    resource_list=resource_list,
                    role=role,
                    level=level,
                    difficulty=difficulty,
                )
                return concept, explanations
            except Exception as e:
                logger.warning(f"Explanation generation failed for '{concept}': {e}")
                return concept, {}

        pairs = await asyncio.gather(
            *[
                _explain_one(concept, resource_list)
                for concept, resource_list in ranked_results.items()
            ]
        )
        return dict(pairs)

    async def _explain_concept_resources(
        self,
        concept: str,
        resource_list: List[Tuple["LearningResource", float]],
        role: str,
        level: str,
        difficulty: str,
    ) -> Dict[str, str]:
        import json as _json

        resource_lines = []
        id_map: Dict[str, "LearningResource"] = {}
        for i, (resource, score) in enumerate(resource_list, start=1):
            id_map[str(i)] = resource
            desc = (resource.description or "")[:150].strip()
            resource_lines.append(
                f'{i}. [{resource.source}] "{resource.title}"'
                + (f" — {desc}" if desc else "")
            )

        context_parts = []
        if role:
            context_parts.append(f"Role: {role}")
        if level:
            context_parts.append(f"Level: {level}")
        if difficulty:
            context_parts.append(f"Difficulty: {difficulty}")
        context_str = " | ".join(context_parts) if context_parts else "general candidate"

        prompt = f"""You are a learning advisor helping a candidate improve after an interview.

Weak concept: {concept}
Candidate context: {context_str}

Retrieved resources (do NOT invent others):
{chr(10).join(resource_lines)}

Task:
1. Re-order these resources from most to least useful for this candidate.
2. Write a 1–2 sentence explanation for each resource explaining why it helps with "{concept}".

Return ONLY valid JSON, no markdown:
{{
  "ranked": [
    {{"id": "1", "explanation": "..."}},
    {{"id": "2", "explanation": "..."}}
  ]
}}

Use only the numbered ids from the list above. Do not add new resources."""

        try:
            response = await self.llm_service.invoke(
                prompt,
                use_cache=True,
                json_mode=True,
                temperature=0.3,
                max_tokens=600,
            )
        except Exception as e:
            logger.warning(f"LLM call failed for concept '{concept}': {e}")
            return {}

        try:
            response = response.strip()
            import re as _re
            response = _re.sub(r"```json\n?|```\n?", "", response)
            match = _re.search(r"\{.*\}", response, _re.DOTALL)
            if not match:
                return {}
            data = _json.loads(match.group(0))
            ranked_items = data.get("ranked", [])
        except Exception as e:
            logger.warning(f"Failed to parse explanation response for '{concept}': {e}")
            return {}

        result: Dict[str, str] = {}
        for item in ranked_items:
            item_id = str(item.get("id", "")).strip()
            explanation = str(item.get("explanation", "")).strip()
            resource = id_map.get(item_id)
            if resource and explanation:
                result[resource.id] = explanation

        return result

    async def retrieve_for_concepts(
        self,
        weak_concepts: List[str],
        role: str = "Software Engineer",
        queries_per_concept: int = 4,
        results_per_query: int = 3,
    ) -> Dict[str, List[LearningResource]]:
        if not weak_concepts:
            return {}
        result = {}
        for concept in weak_concepts:
            try:
                queries = await self.query_generator.generate_queries(
                    concept, role=role, count=queries_per_concept
                )
                if not queries:
                    result[concept] = []
                    continue
                raw = await self._fetch_for_queries(queries, results_per_query)
                normalized = await self._normalize_resources(raw)
                result[concept] = self._deduplicate_by_url(normalized)
                logger.info(
                    f"Retrieved {len(result[concept])} unique resources for '{concept}'"
                )
            except Exception as e:
                logger.error(f"Error retrieving for '{concept}': {e}")
                result[concept] = []
        return result

    async def _fetch_for_queries(
        self,
        queries: List[str],
        limit_per_query: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        async def _fetch_one(fetcher: BaseFetcher, query: str) -> Tuple[str, str, List[Dict[str, Any]]]:
            async with self._http_semaphore:
                try:
                    hits = await fetcher.search(query, limit=limit_per_query)
                    return fetcher.source, query, hits
                except Exception as e:
                    logger.warning(f"Fetch failed '{query}' from {fetcher.source}: {e}")
                    return fetcher.source, query, []

        tasks = [
            _fetch_one(fetcher, query)
            for fetcher in self.fetchers
            for query in queries
        ]
        results = await asyncio.gather(*tasks)

        raw_by_source: Dict[str, List[Dict[str, Any]]] = {}
        for source, _query, hits in results:
            raw_by_source.setdefault(source, []).extend(hits)

        for source, items in raw_by_source.items():
            logger.info(f"Fetched {len(items)} results from {source}")

        return raw_by_source

    async def _normalize_resources(
        self,
        raw_by_source: Dict[str, List[Dict]],
    ) -> List[LearningResource]:
        normalized: List[LearningResource] = []
        normalizer_map = {
            "youtube": ResourceNormalizer.normalize_youtube,
            "geeksforgeeks": ResourceNormalizer.normalize_geeksforgeeks,
        }
        for source, raw_resources in raw_by_source.items():
            try:
                fn = normalizer_map.get(source, ResourceNormalizer.normalize_generic)
                batch = ResourceNormalizer.normalize_batch(raw_resources, source, fn)
                normalized.extend(batch)
                logger.info(f"Normalized {len(batch)} resources from {source}")
            except Exception as e:
                logger.error(f"Normalization failed for {source}: {e}")
        return normalized

    def _deduplicate_by_url(
        self, resources: List[LearningResource]
    ) -> List[LearningResource]:
        seen: set = set()
        deduped: List[LearningResource] = []
        for r in resources:
            key = r.url.lower().strip()
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        logger.info(f"Deduplicated {len(resources)} → {len(deduped)} unique resources")
        return deduped

    def add_fetcher(self, fetcher: BaseFetcher) -> None:
        if any(f.source == fetcher.source for f in self.fetchers):
            logger.warning(f"Fetcher for {fetcher.source} already exists. Replacing.")
            self.fetchers = [f for f in self.fetchers if f.source != fetcher.source]
        self.fetchers.append(fetcher)
        logger.info(f"Added fetcher for source: {fetcher.source}")

    def list_fetchers(self) -> List[str]:
        return [f.source for f in self.fetchers]
