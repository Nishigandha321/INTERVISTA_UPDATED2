# services/learning_resources/query_generator.py
# Generate context-aware search queries from weak concepts

import json
import re
from typing import List
from core.llm.llm_service import LLMService
from core.prompts.prompt_manager import PromptManager
from utils.logger import Logger

logger = Logger(__name__)


class SearchQueryGenerator:
    """
    Generate context-aware search queries from weak concepts.

    Uses the LLM to create diverse, targeted search queries that reflect
    the candidate's role, experience level, target company, and how hard
    the interview was — so retrieved resources are appropriately pitched.
    """

    def __init__(self, llm_service: LLMService, prompt_manager: PromptManager):
        self.llm_service = llm_service
        self.prompt_manager = prompt_manager

    async def generate_queries(
        self,
        concept: str,
        role: str = "Software Engineer",
        count: int = 4,
        level: str = "",
        company: str = "",
        difficulty: str = "",
    ) -> List[str]:
        """
        Generate search queries for a weak concept.

        Args:
            concept:    Weak topic (e.g. "Binary Search Trees").
            role:       Job role (e.g. "Senior Backend Engineer").
            count:      Number of queries to generate (1–10).
            level:      Experience level (e.g. "junior", "senior").
            company:    Target company, if known (e.g. "Google").
            difficulty: Interview difficulty derived from score
                        ("beginner", "intermediate", "advanced").

        Returns:
            List of search query strings optimised for resource discovery.
        """
        if not concept or not concept.strip():
            logger.warning("Empty concept provided to query generator")
            return []

        concept = " ".join(concept.split())
        count = max(1, min(count, 10))
        # These are search terms, not interview content. Keep them deterministic:
        # this avoids up to one extra LLM call per weakness and prevents unrelated
        # company/role names from diluting a candidate-specific skill gap.
        queries = [
            f"{concept} fundamentals",
            f"{concept} concepts and examples",
            f"{concept} beginner tutorial",
            f"{concept} common mistakes",
        ]
        logger.info(f"Built {min(count, len(queries))} focused resource queries for '{concept}'")
        return queries[:count]

    async def generate_queries_batch(
        self,
        concepts: List[str],
        role: str = "Software Engineer",
        count: int = 4,
        level: str = "",
        company: str = "",
        difficulty: str = "",
    ) -> dict:
        """
        Generate queries for multiple concepts in sequence.

        Returns:
            Dict mapping concept → [queries].
        """
        result = {}
        for concept in concepts:
            try:
                queries = await self.generate_queries(
                    concept, role=role, count=count,
                    level=level, company=company, difficulty=difficulty,
                )
                result[concept] = queries
            except Exception as e:
                logger.warning(f"Failed to generate queries for '{concept}': {e}")
                result[concept] = self._get_fallback_queries(
                    concept, role, count, level, difficulty
                )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        concept: str,
        role: str,
        count: int,
        level: str,
        company: str,
        difficulty: str,
    ) -> str:
        """Build the LLM prompt, injecting all available context."""

        # Only include context lines that have meaningful values
        context_lines = []
        if level:
            context_lines.append(f"Experience Level: {level}")
        if company:
            context_lines.append(f"Target Company: {company}")
        if difficulty:
            context_lines.append(f"Interview Difficulty: {difficulty}")

        context_block = (
            "\n".join(context_lines) + "\n" if context_lines else ""
        )

        # Tailor the instruction based on difficulty so the LLM can vary depth
        depth_hint = ""
        if difficulty in ("beginner", "easy"):
            depth_hint = "Favour introductory and conceptual resources."
        elif difficulty in ("advanced", "hard"):
            depth_hint = (
                "Favour advanced, system-design-level, and interview-focused resources. "
                f"Where relevant, include queries specific to {company or 'top tech companies'}."
            )
        else:
            depth_hint = "Mix beginner, intermediate, and interview-prep resources."

        return f"""You are an expert learning resource discovery system.

Your task: Generate {count} diverse, high-quality search queries for a weak concept.

Weak Concept: {concept}
Target Role: {role}
{context_block}
Guidelines:
1. Generate exactly {count} search queries.
2. Vary the approach: conceptual explanations, coding walkthroughs, interview prep, best practices.
3. {depth_hint}
4. Include role-specific and level-specific language where relevant.
5. Keep each query concise (under 12 words) and suited to YouTube or article search.

Return ONLY valid JSON — no markdown, no explanation:
{{
  "queries": [
    "query1",
    "query2",
    "query3",
    "query4"
  ]
}}"""

    def _parse_response(self, response: str) -> List[str]:
        """Parse LLM JSON response into a list of query strings."""
        try:
            response = response.strip()
            response = re.sub(r"```json\n?", "", response)
            response = re.sub(r"```\n?", "", response)

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in LLM response")
                return []

            json_str = json_match.group(0)
            json_str = re.sub(r",\s*}", "}", json_str)
            json_str = re.sub(r",\s*]", "]", json_str)

            data = json.loads(json_str)

            queries = (
                data.get("queries", [])
                if isinstance(data, dict)
                else data if isinstance(data, list)
                else []
            )

            return [
                str(q).strip()
                for q in queries
                if str(q).strip() and len(str(q).strip()) < 200
            ]

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return []
        except Exception as e:
            logger.error(f"Response parsing error: {e}")
            return []

    def _get_fallback_queries(
        self,
        concept: str,
        role: str,
        count: int,
        level: str,
        difficulty: str,
    ) -> List[str]:
        """
        Heuristic fallback when the LLM call fails.

        Incorporates role and difficulty so fallbacks are still context-aware.
        """
        role_tag = f"{role} " if role else ""
        level_tag = f"{level} " if level else ""

        candidates = [
            f"{concept} tutorial",
            f"{concept} interview questions {role_tag}".strip(),
            f"{level_tag}{concept} explained".strip(),
            f"{concept} best practices",
            f"{concept} with examples",
            f"{concept} implementation guide",
            f"how to learn {concept}",
            f"{concept} for {role_tag or 'software engineers'}".strip(),
        ]

        # For advanced difficulty, swap generic tutorials for deeper content
        if difficulty in ("advanced", "hard"):
            candidates[0] = f"{concept} system design"
            candidates[2] = f"{concept} advanced {role_tag}interview".strip()

        seen: set = set()
        unique: List[str] = []
        for q in candidates:
            key = q.lower()
            if key not in seen:
                seen.add(key)
                unique.append(q)

        logger.info(f"Using {min(count, len(unique))} fallback queries for: {concept}")
        return unique[:count]
