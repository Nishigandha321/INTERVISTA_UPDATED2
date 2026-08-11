# services/learning_resources/ranker.py
# Semantic ranking of learning resources

from typing import List, Tuple, Dict, Any, Optional
import math
from .normalizer import LearningResource
from utils.logger import Logger

logger = Logger(__name__)


class SemanticRanker:
    """
    Rank learning resources by multiple relevance signals.
    
    Scoring formula:
    ================
    final_score = 0.40 * semantic_similarity
               + 0.20 * source_authority
               + 0.20 * popularity
               + 0.10 * freshness
               + 0.10 * duration_relevance
    
    Each component is normalized to 0.0-1.0 range.
    """
    
    # Source authority scores (higher = more trustworthy)
    SOURCE_AUTHORITY = {
        "coursera": 0.95,
        "edx": 0.95,
        "microsoft_learn": 0.92,
        "aws_skill_builder": 0.90,
        "google_cloud_skills": 0.90,
        "geeksforgeeks": 0.85,
        "youtube": 0.80,
        "linkedin_learning": 0.80,
        "udemy": 0.70,
        "medium": 0.60,
    }
    
    def __init__(self, embedder=None):
        """
        Initialize ranker with optional embedder.
        
        Args:
            embedder: TextEmbedder for semantic similarity
                     If None, semantic ranking is skipped
        """
        self.embedder = embedder
        logger.info(
            f"SemanticRanker initialized "
            f"(semantic scoring: {'enabled' if embedder else 'disabled'})"
        )
    
    def rank_resources(
        self,
        weak_concept: str,
        resources: List[LearningResource],
        role: str = "Software Engineer",
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[LearningResource, float]]:
        """
        Rank resources by relevance score.
        
        Args:
            weak_concept: The concept being learned (e.g., "Binary Trees")
            resources: List of resources to rank
            role: Job role for context
            weights: Custom weights for scoring components (optional)
        
        Returns:
            List of (resource, score) tuples sorted by score descending
            Score ranges from 0.0 to 1.0
        
        Example:
            ranked = ranker.rank_resources(
                weak_concept="Database Sharding",
                resources=[resource1, resource2, resource3],
                role="Senior Backend Engineer"
            )
            
            # Returns: [
            #   (resource2, 0.92),
            #   (resource1, 0.87),
            #   (resource3, 0.71),
            # ]
        """
        
        if not resources:
            logger.warning("No resources to rank")
            return []
        
        # Use default weights if not provided
        if weights is None:
            weights = {
                "semantic": 0.40,
                "authority": 0.20,
                "popularity": 0.20,
                "freshness": 0.10,
                "duration": 0.10,
            }
        
        # Validate weights sum to 1.0
        weight_sum = sum(weights.values())
        if not (0.99 <= weight_sum <= 1.01):
            logger.warning(f"Weights sum to {weight_sum}, normalizing...")
            for key in weights:
                weights[key] /= weight_sum
        
        # Calculate scores for all resources
        scored_resources = []
        
        for resource in resources:
            try:
                score = self._calculate_resource_score(
                    weak_concept=weak_concept,
                    resource=resource,
                    weights=weights,
                )
                scored_resources.append((resource, score))
            except Exception as e:
                logger.warning(f"Scoring failed for {resource.title}: {str(e)}")
                # Assign minimum score on error
                scored_resources.append((resource, 0.0))
        
        # Sort by score descending
        ranked = sorted(scored_resources, key=lambda x: x[1], reverse=True)
        
        logger.info(
            f"Ranked {len(ranked)} resources for concept: {weak_concept}\n"
            f"  Top score: {ranked[0][1]:.2f}\n"
            f"  Avg score: {sum(s for _, s in ranked) / len(ranked):.2f}"
        )
        
        return ranked
    
    def _calculate_resource_score(
        self,
        weak_concept: str,
        resource: LearningResource,
        weights: Dict[str, float],
    ) -> float:
        """Calculate comprehensive score for a single resource."""
        
        components = {}
        
        # 1. Semantic Similarity (40%)
        if self.embedder:
            components["semantic"] = self._semantic_similarity(weak_concept, resource)
        else:
            # Fallback: keyword matching if embedder unavailable — ranking degrades
            # from semantic to lexical overlap (logged once per rank_resources call).
            if not getattr(self, "_keyword_fallback_logged", False):
                logger.warning(
                    "SemanticRanker: embedder is None — using keyword fallback; "
                    "ranking quality may degrade vs semantic similarity"
                )
                self._keyword_fallback_logged = True
            components["semantic"] = self._keyword_similarity(weak_concept, resource)
        
        # 2. Source Authority (20%)
        components["authority"] = self._get_source_authority(resource.source)
        
        # 3. Popularity (20%)
        # NOTE: YouTube popularity uses view-count-derived scores while GFG uses a
        # flat default (~0.7) — these signals are not directly comparable across sources.
        # Normalizing per-source popularity is a future improvement.
        components["popularity"] = self._normalize_popularity(
            resource.popularity_score,
            resource.rating,
            resource.rating_count,
        )
        
        # 4. Freshness (10%)
        components["freshness"] = self._calculate_freshness(resource.published_date)
        
        # 5. Duration Relevance (10%)
        components["duration"] = self._duration_relevance(resource.duration_seconds)
        
        # Weighted sum
        final_score = sum(
            components.get(key, 0.0) * weight
            for key, weight in weights.items()
        )
        
        # Clamp to 0.0-1.0
        return max(0.0, min(1.0, final_score))
    
    def _semantic_similarity(
        self,
        weak_concept: str,
        resource: LearningResource
    ) -> float:
        """
        Calculate semantic similarity using embeddings.
        
        Compares embedding of weak concept with resource's title + description.
        """
        
        if not self.embedder:
            return 0.0
        
        try:
            # Generate embeddings
            concept_embedding = self.embedder.embed(weak_concept)
            resource_text = resource.to_embedding_text()
            resource_embedding = self.embedder.embed(resource_text)
            
            # Cosine similarity
            similarity = self._cosine_similarity(concept_embedding, resource_embedding)
            
            # Normalize to 0.0-1.0 (cosine ranges -1 to 1)
            normalized = (similarity + 1.0) / 2.0
            
            return max(0.0, min(1.0, normalized))
        
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {str(e)}")
            return 0.5  # Neutral score on error
    
    def _keyword_similarity(
        self,
        weak_concept: str,
        resource: LearningResource
    ) -> float:
        """
        Fallback keyword-based similarity when embedder unavailable.
        
        Simple heuristic: how much does resource match the concept?
        """
        
        concept_words = set(weak_concept.lower().split())
        
        # Check concept in title
        title_match = sum(
            1 for word in concept_words
            if word in resource.title.lower()
        ) / max(len(concept_words), 1)
        
        # Check concept in description
        desc_match = sum(
            1 for word in concept_words
            if word in resource.description.lower()
        ) / max(len(concept_words), 1)
        
        # Check concept in tags
        tag_match = sum(
            1 for word in concept_words
            if any(word in tag.lower() for tag in resource.tags)
        ) / max(len(concept_words), 1)
        
        # Weighted average: title (50%) > description (30%) > tags (20%)
        score = 0.50 * title_match + 0.30 * desc_match + 0.20 * tag_match
        
        return max(0.0, min(1.0, score))
    
    def _get_source_authority(self, source: str) -> float:
        """Get authority score for source."""
        source_lower = source.lower().strip()
        return self.SOURCE_AUTHORITY.get(source_lower, 0.50)
    
    def _normalize_popularity(
        self,
        popularity_score: float,
        rating: float,
        rating_count: int,
    ) -> float:
        """
        Normalize popularity metrics to 0.0-1.0.
        
        Combines:
        - Popularity score (0.0-1.0)
        - User rating (0.0-5.0)
        - Rating count (more ratings = more trustworthy)
        """
        
        # Normalize rating to 0.0-1.0
        rating_normalized = max(0.0, min(1.0, rating / 5.0))
        
        # Weight rating by rating count (more ratings = higher confidence)
        # Log scale: 1-10 ratings = low, 100+ = high
        rating_confidence = min(1.0, math.log(max(1, rating_count)) / math.log(100))
        
        # Weighted average: popularity (50%) > rating (50% but weighted by confidence)
        score = (
            0.50 * popularity_score +
            0.50 * (rating_normalized * rating_confidence)
        )
        
        return max(0.0, min(1.0, score))
    
    def _calculate_freshness(self, published_date: Optional[str]) -> float:
        """
        Calculate freshness score based on publication date.
        
        Recent content (< 6 months) scores higher.
        Older content (> 2 years) scores lower.
        """
        
        if not published_date:
            return 0.5  # Neutral score if date unknown
        
        try:
            from datetime import datetime, timezone, timedelta
            
            # Parse date (format: "2024-01-15")
            pub_date = datetime.fromisoformat(published_date)
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            
            days_old = (now - pub_date).days
            
            # Scoring curve:
            # 0 days old = 1.0 (perfect freshness)
            # 180 days (6 months) = 0.8
            # 365 days (1 year) = 0.6
            # 730 days (2 years) = 0.4
            # 1095 days (3 years) = 0.2
            # 1825+ days (5+ years) = 0.0
            
            if days_old <= 0:
                return 1.0
            elif days_old <= 180:
                return 1.0 - (days_old / 180) * 0.2
            elif days_old <= 365:
                return 0.8 - ((days_old - 180) / 185) * 0.2
            elif days_old <= 730:
                return 0.6 - ((days_old - 365) / 365) * 0.2
            elif days_old <= 1095:
                return 0.4 - ((days_old - 730) / 365) * 0.2
            else:
                return 0.0
        
        except Exception as e:
            logger.warning(f"Freshness calculation failed: {str(e)}")
            return 0.5
    
    def _duration_relevance(self, duration_seconds: int) -> float:
        """
        Score based on resource duration (time to learn).
        
        Optimal range: 15-45 minutes for interview prep.
        Too short (< 5 min) or too long (> 2 hours) scores lower.
        """
        
        if duration_seconds <= 0:
            return 0.5  # Neutral score if duration unknown
        
        duration_minutes = duration_seconds / 60
        
        # Scoring curve:
        # 0-5 min = 0.3 (too brief for depth)
        # 5-15 min = 0.7 (intro/quick review)
        # 15-45 min = 1.0 (optimal learning time)
        # 45-120 min = 0.8 (comprehensive but long)
        # 120+ min = 0.4 (very long, requires commitment)
        
        if duration_minutes < 5:
            return 0.3
        elif duration_minutes <= 15:
            return 0.3 + (duration_minutes - 5) / 10 * 0.4  # 0.3 -> 0.7
        elif duration_minutes <= 45:
            return 0.7 + (duration_minutes - 15) / 30 * 0.3  # 0.7 -> 1.0
        elif duration_minutes <= 120:
            return 0.8 - (duration_minutes - 45) / 75 * 0.2  # 0.8 -> 0.6
        else:
            return 0.4  # Too long
    
    @staticmethod
    def _cosine_similarity(vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors."""
        
        try:
            import numpy as np
            
            # Ensure numpy arrays
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            # Dot product
            dot_product = np.dot(v1, v2)
            
            # Magnitudes
            magnitude1 = np.linalg.norm(v1)
            magnitude2 = np.linalg.norm(v2)
            
            # Avoid division by zero
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return float(dot_product / (magnitude1 * magnitude2))
        
        except Exception as e:
            logger.warning(f"Cosine similarity calculation failed: {str(e)}")
            return 0.0
    
    def filter_by_threshold(
        self,
        ranked_resources: List[Tuple[LearningResource, float]],
        threshold: float = 0.50,
    ) -> List[Tuple[LearningResource, float]]:
        """
        Filter resources by minimum score threshold.
        
        Args:
            ranked_resources: Already-ranked resources
            threshold: Minimum score to keep (0.0-1.0)
        
        Returns:
            Filtered list keeping only resources above threshold
        """
        
        filtered = [
            (resource, score)
            for resource, score in ranked_resources
            if score >= threshold
        ]
        
        logger.info(
            f"Filtered {len(ranked_resources)} → {len(filtered)} resources "
            f"(threshold: {threshold})"
        )
        
        return filtered
    
    def get_top_k(
        self,
        ranked_resources: List[Tuple[LearningResource, float]],
        k: int = 5,
    ) -> List[LearningResource]:
        """
        Get top K resources by score.
        
        Args:
            ranked_resources: Already-ranked resources
            k: Number of resources to return
        
        Returns:
            Top K resources (scores removed)
        """
        
        return [resource for resource, _ in ranked_resources[:k]]
