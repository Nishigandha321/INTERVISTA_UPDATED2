# rag/vector_store.py
# Vector store for embeddings and semantic search

import os
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

from config.settings import Settings
from utils.logger import Logger

logger = Logger(__name__)


class NumpyIndex:
    """Fallback index for environments without FAISS."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = np.zeros((0, dimension), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def add(self, vectors: np.ndarray):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if self.vectors.size == 0:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

    def search(self, queries: np.ndarray, k: int):
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if self.ntotal == 0:
            return np.empty((queries.shape[0], 0), dtype=np.float32), np.full((queries.shape[0], 0), -1, dtype=np.int64)

        # Use squared L2 distances for ranking
        diffs = self.vectors[np.newaxis, :, :] - queries[:, np.newaxis, :]
        distances = np.sum(diffs * diffs, axis=2)
        k = min(k, self.ntotal)
        indices = np.argpartition(distances, k - 1, axis=1)[:, :k]
        ordered = np.take_along_axis(distances, indices, axis=1)
        sort_order = np.argsort(ordered, axis=1)
        sorted_indices = np.take_along_axis(indices, sort_order, axis=1)
        sorted_distances = np.take_along_axis(ordered, sort_order, axis=1)
        return sorted_distances, sorted_indices


class TextEmbedder:
    """Wrapper for text embeddings (using sentence-transformers via FAISS)."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize embedder."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            # Use get_embedding_dimension() (replaces deprecated get_sentence_embedding_dimension())
            self.embedding_dim = self.model.get_embedding_dimension()
            logger.info(f"TextEmbedder initialized: {model_name} (dim={self.embedding_dim})")
        except ImportError:
            logger.warning("sentence-transformers not installed. Using mock embedder.")
            self.model = None
            self.embedding_dim = 384
    
    def embed(self, text: str) -> np.ndarray:
        """Convert text to embedding."""
        if self.model is None:
            # Mock embedding for when sentence-transformers not available
            return np.random.randn(self.embedding_dim).astype(np.float32)
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Convert multiple texts to embeddings."""
        if self.model is None:
            return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(np.float32)


class Document:
    """Represents a document in the vector store."""
    
    def __init__(self, doc_id: str, content: str, metadata: Dict[str, Any] = None):
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata or {}
        self.embedding = None
    
    def to_dict(self) -> Dict:
        """Serialize document."""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata
        }


class VectorStore:
    """
    Vector database backed by FAISS with multi-collection support.
    
    Stores documents with embeddings for semantic search.
    
    Collections:
    - "interview_tips": Interview preparation content
    - "learning_resources": Learning resources (YouTube, GFG, Coursera, etc.)
    - "company_knowledge": Company-specific knowledge
    - "default": Default collection (backward compatibility)
    """
    
    DEFAULT_COLLECTION = "default"
    RESOURCE_COLLECTION = "learning_resources"
    RESOURCE_TTL_SECONDS = 3600  # 1 hour TTL for cached resource embeddings
    
    def __init__(self, embedding_dim: int = 384, index_type: str = "faiss"):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.faiss_available = faiss is not None and index_type == "faiss"
        self.embedder = TextEmbedder()
        
        if not self.faiss_available and index_type == "faiss":
            logger.warning("FAISS not available. Falling back to numpy index for vector search.")
        
        # Multi-collection support
        self.collections: Dict[str, Dict[str, Any]] = {}
        self._create_collection(self.DEFAULT_COLLECTION)
        self._create_collection(self.RESOURCE_COLLECTION)
        
        logger.info(
            f"VectorStore initialized with collections support "
            f"(dim={embedding_dim}, type={index_type}, faiss_available={self.faiss_available})"
        )
    
    # ------------------------------------------------------------------
    # Backward-compat properties: proxy self.index / self.documents /
    # self.doc_id_to_index to the DEFAULT collection so legacy callers
    # (add_document, search, etc.) keep working unchanged.
    # ------------------------------------------------------------------
    
    @property
    def index(self):
        return self.collections[self.DEFAULT_COLLECTION]["index"]
    
    @property
    def documents(self):
        return self.collections[self.DEFAULT_COLLECTION]["documents"]
    
    @property
    def doc_id_to_index(self):
        return self.collections[self.DEFAULT_COLLECTION]["doc_id_to_index"]
    
    def _make_index(self):
        if self.faiss_available:
            return faiss.IndexFlatL2(self.embedding_dim)
        return NumpyIndex(self.embedding_dim)
    
    def _init_index(self):
        """Re-initialize the default collection's index."""
        self.collections[self.DEFAULT_COLLECTION]["index"] = self._make_index()
        logger.debug("Vector index created (default collection)")
    
    def _create_collection(self, collection_name: str):
        """Create a new collection with its own index."""
        if collection_name in self.collections:
            logger.warning(f"Collection '{collection_name}' already exists")
            return
        
        self.collections[collection_name] = {
            "documents": {},
            "index": self._make_index(),
            "doc_id_to_index": {},
            "created_at": __import__("datetime").datetime.utcnow().isoformat(),
        }
        
        logger.info(f"Collection created: {collection_name}")
    
    def _get_collection(self, collection_name: str = None):
        """Get collection data, creating if needed."""
        if collection_name is None:
            collection_name = self.DEFAULT_COLLECTION
        
        if collection_name not in self.collections:
            self._create_collection(collection_name)
        
        return self.collections[collection_name]
    
    def list_collections(self) -> List[str]:
        """Get all collection names."""
        return list(self.collections.keys())
    
    def collection_stats(self, collection_name: str = None) -> Dict[str, Any]:
        """Get statistics for a collection."""
        collection = self._get_collection(collection_name)
        return {
            "name": collection_name or self.DEFAULT_COLLECTION,
            "documents": len(collection["documents"]),
            "created_at": collection.get("created_at"),
        }

    # ------------------------------------------------------------------
    # Legacy methods (operate on DEFAULT collection via properties)
    # ------------------------------------------------------------------
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add document to vector store (default collection).
        
        Args:
            doc_id: Unique document ID
            content: Document text content
            metadata: Optional metadata dict
        
        Returns:
            Document ID
        """
        
        # Create document
        doc = Document(doc_id, content, metadata)
        
        # Generate embedding
        embedding = self.embedder.embed(content)
        doc.embedding = embedding
        
        # Add to FAISS index
        index_pos = self.index.ntotal
        self.index.add(np.array([embedding]))
        self.doc_id_to_index[doc_id] = index_pos
        
        # Store document
        self.documents[doc_id] = doc
        
        logger.debug(f"Document added: {doc_id}")
        return doc_id
    
    def add_documents_batch(self, documents: List[Tuple[str, str, Dict]] = None, documents_list: List[Dict] = None) -> List[str]:
        """
        Add multiple documents efficiently (default collection).
        
        Args:
            documents: List of (doc_id, content, metadata) tuples
            documents_list: List of {"doc_id": ..., "content": ..., "metadata": ...}
        
        Returns:
            List of added doc_ids
        """
        
        if documents_list is None:
            documents_list = []
        
        if documents:
            documents_list = [(d[0], d[1], d[2] if len(d) > 2 else {}) for d in documents]
        
        if not documents_list:
            return []
        
        doc_ids = []
        embeddings = []
        
        # Prepare embeddings
        texts = [doc[1] for doc in documents_list]
        batch_embeddings = self.embedder.embed_batch(texts)
        
        # Add documents
        for i, (doc_id, content, metadata) in enumerate(documents_list):
            doc = Document(doc_id, content, metadata)
            doc.embedding = batch_embeddings[i]
            
            self.doc_id_to_index[doc_id] = self.index.ntotal + i
            self.documents[doc_id] = doc
            doc_ids.append(doc_id)
            embeddings.append(batch_embeddings[i])
        
        # Add all embeddings to index at once
        self.index.add(np.array(embeddings))
        
        logger.info(f"Batch added: {len(doc_ids)} documents")
        return doc_ids
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Semantic search in vector store (default collection).
        
        Args:
            query: Search query text
            k: Number of results to return
        
        Returns:
            List of (doc_id, similarity_score) tuples
        """
        
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Embed query
        query_embedding = self.embedder.embed(query)
        
        # Search FAISS index (returns distances)
        distances, indices = self.index.search(np.array([query_embedding]), k)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            
            # Find doc_id for this index position
            doc_id = self._get_doc_id_by_index(idx)
            if doc_id:
                # Convert L2 distance to similarity score (0-100)
                similarity = 100 / (1 + distance)
                results.append((doc_id, similarity))
        
        logger.debug(f"Search returned {len(results)} results")
        return results
    
    def _get_doc_id_by_index(self, index_pos: int) -> Optional[str]:
        """Get doc_id by FAISS index position (default collection)."""
        for doc_id, idx_pos in self.doc_id_to_index.items():
            if idx_pos == index_pos:
                return doc_id
        return None
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve document by ID (default collection)."""
        return self.documents.get(doc_id)
    
    def delete_document(self, doc_id: str) -> bool:
        """Remove document (marks as deleted, default collection)."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            logger.debug(f"Document deleted: {doc_id}")
            return True
        
        return False
    
    def size(self) -> int:
        """Total documents in store (default collection)."""
        return len(self.documents)
    
    def stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        resource_col = self._get_collection(self.RESOURCE_COLLECTION)
        return {
            "total_documents": self.size(),
            "index_type": self.index_type,
            "embedding_dimension": self.embedding_dim,
            "faiss_index_size": self.index.ntotal,
            "resource_collection_size": len(resource_col["documents"]),
            "collections": self.list_collections(),
        }
    
    def save(self, path: str):
        """Save vector store to disk."""
        data = {
            "documents": {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()},
            "index_type": self.index_type,
            "embedding_dim": self.embedding_dim
        }
        
        with open(path, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"VectorStore saved to {path}")
    
    def load(self, path: str):
        """Load vector store from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        for doc_id, doc_data in data.get("documents", {}).items():
            self.add_document(
                doc_id=doc_id,
                content=doc_data["content"],
                metadata=doc_data.get("metadata", {})
            )
        
        logger.info(f"VectorStore loaded from {path}")
    
    def clear(self):
        """Clear all documents (default collection)."""
        self.documents.clear()
        self.doc_id_to_index.clear()
        self._init_index()
        logger.info("VectorStore cleared")
    
    # ==================================================================
    # LEARNING RESOURCES COLLECTION — Phase 2
    # ==================================================================
    
    def add_resource(
        self,
        resource_id: str,
        embedding_text: str,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Add a single learning resource to the learning_resources collection.
        
        Args:
            resource_id: Unique resource ID (e.g., "youtube_abc123")
            embedding_text: Text to embed (title + description + tags)
            metadata: Resource metadata dict (full LearningResource.to_dict())
        
        Returns:
            Resource ID
        """
        from datetime import datetime, timezone
        
        col = self._get_collection(self.RESOURCE_COLLECTION)
        
        # Skip if already cached
        if resource_id in col["documents"]:
            logger.debug(f"Resource already cached: {resource_id}")
            return resource_id
        
        doc = Document(resource_id, embedding_text, metadata or {})
        embedding = self.embedder.embed(embedding_text)
        doc.embedding = embedding
        
        # Track insertion time for TTL eviction
        doc.metadata["_cached_at"] = datetime.now(timezone.utc).isoformat()
        
        index_pos = col["index"].ntotal
        col["index"].add(np.array([embedding]))
        col["doc_id_to_index"][resource_id] = index_pos
        col["documents"][resource_id] = doc
        
        logger.debug(f"Resource added to collection: {resource_id}")
        return resource_id
    
    def add_resources_batch(
        self,
        resources: List[Tuple[str, str, Dict]],
    ) -> List[str]:
        """
        Batch-add learning resources to the learning_resources collection.
        
        Args:
            resources: List of (resource_id, embedding_text, metadata) tuples
        
        Returns:
            List of added resource IDs
        """
        from datetime import datetime, timezone
        
        if not resources:
            return []
        
        col = self._get_collection(self.RESOURCE_COLLECTION)
        now_iso = datetime.now(timezone.utc).isoformat()
        
        # Filter out already-cached resources
        new_resources = [
            (rid, text, meta)
            for rid, text, meta in resources
            if rid not in col["documents"]
        ]
        
        if not new_resources:
            logger.debug("All resources already cached, skipping batch add")
            return [r[0] for r in resources]
        
        # Generate embeddings in batch
        texts = [text for _, text, _ in new_resources]
        batch_embeddings = self.embedder.embed_batch(texts)
        
        added_ids = []
        embeddings_to_add = []
        
        for i, (resource_id, embedding_text, metadata) in enumerate(new_resources):
            doc = Document(resource_id, embedding_text, metadata or {})
            doc.embedding = batch_embeddings[i]
            doc.metadata["_cached_at"] = now_iso
            
            col["doc_id_to_index"][resource_id] = col["index"].ntotal + i
            col["documents"][resource_id] = doc
            added_ids.append(resource_id)
            embeddings_to_add.append(batch_embeddings[i])
        
        # Add all embeddings to FAISS index at once
        if embeddings_to_add:
            col["index"].add(np.array(embeddings_to_add))
        
        logger.info(
            f"Batch added {len(added_ids)} resources to learning_resources collection "
            f"(total: {len(col['documents'])})"
        )
        return added_ids
    
    def search_resources(
        self,
        query: str,
        k: int = 10,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Semantic search in the learning_resources collection only.
        
        Args:
            query: Search query text
            k: Number of results to return
        
        Returns:
            List of (resource_id, similarity_score, metadata) tuples
        """
        col = self._get_collection(self.RESOURCE_COLLECTION)
        
        if col["index"].ntotal == 0:
            logger.debug("Resource collection is empty")
            return []
        
        query_embedding = self.embedder.embed(query)
        
        effective_k = min(k, col["index"].ntotal)
        distances, indices = col["index"].search(
            np.array([query_embedding]), effective_k
        )
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            
            # Find resource_id for this index position
            resource_id = None
            for rid, idx_pos in col["doc_id_to_index"].items():
                if idx_pos == idx:
                    resource_id = rid
                    break
            
            if resource_id and resource_id in col["documents"]:
                similarity = 100 / (1 + distance)
                metadata = col["documents"][resource_id].metadata
                results.append((resource_id, similarity, metadata))
        
        logger.debug(f"Resource search returned {len(results)} results")
        return results
    
    def evict_stale_resources(self) -> int:
        """
        Remove resources older than RESOURCE_TTL_SECONDS from the cache.
        
        Note: FAISS doesn't support deletion, so we rebuild the index.
        This is acceptable because the resource collection is small
        (typically < 100 items between evictions).
        
        Returns:
            Number of evicted resources
        """
        from datetime import datetime, timezone
        
        col = self._get_collection(self.RESOURCE_COLLECTION)
        now = datetime.now(timezone.utc)
        stale_ids = []
        
        for doc_id, doc in col["documents"].items():
            cached_at_str = doc.metadata.get("_cached_at")
            if not cached_at_str:
                continue
            try:
                cached_at = datetime.fromisoformat(cached_at_str)
                if cached_at.tzinfo is None:
                    cached_at = cached_at.replace(tzinfo=timezone.utc)
                age_seconds = (now - cached_at).total_seconds()
                if age_seconds > self.RESOURCE_TTL_SECONDS:
                    stale_ids.append(doc_id)
            except Exception:
                continue
        
        if not stale_ids:
            return 0
        
        # Remove stale documents
        for doc_id in stale_ids:
            col["documents"].pop(doc_id, None)
            col["doc_id_to_index"].pop(doc_id, None)
        
        # Rebuild FAISS index from remaining documents
        col["index"] = faiss.IndexFlatL2(self.embedding_dim)
        new_id_map = {}
        embeddings = []
        
        for doc_id, doc in col["documents"].items():
            if doc.embedding is not None:
                new_id_map[doc_id] = len(embeddings)
                embeddings.append(doc.embedding)
        
        if embeddings:
            col["index"].add(np.array(embeddings))
        
        col["doc_id_to_index"] = new_id_map
        
        logger.info(
            f"Evicted {len(stale_ids)} stale resources "
            f"(remaining: {len(col['documents'])})"
        )
        return len(stale_ids)
    
    def resource_collection_size(self) -> int:
        """Get number of resources in the learning_resources collection."""
        col = self._get_collection(self.RESOURCE_COLLECTION)
        return len(col["documents"])
