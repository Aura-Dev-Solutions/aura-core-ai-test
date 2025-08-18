"""
Vector search implementation using FAISS for efficient similarity search.
Provides indexing and search capabilities for document embeddings.
"""

import asyncio
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from src.core.exceptions import VectorDatabaseError, SearchError, IndexError
from src.core.config import settings
from src.core.logging import LoggerMixin, log_performance
from src.core.models import SearchQuery, SearchResult


class FAISSVectorStore(LoggerMixin):
    """
    FAISS-based vector store for efficient similarity search.
    
    Supports multiple index types:
    - Flat: Exact search, good for small datasets
    - IVF: Inverted file index, good for medium datasets
    - HNSW: Hierarchical NSW, good for large datasets
    """
    
    def __init__(
        self,
        dimension: int,
        index_type: str = "Flat",
        metric: str = "L2",
        storage_path: Optional[Path] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of vectors
            index_type: Type of FAISS index ('Flat', 'IVF', 'HNSW')
            metric: Distance metric ('L2', 'IP' for inner product)
            storage_path: Path to store the index
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.storage_path = storage_path or settings.vector_db_path
        
        self.index = None
        self.metadata_store = {}  # Store document metadata
        self.id_to_index = {}     # Map document IDs to index positions
        self.index_to_id = {}     # Map index positions to document IDs
        self.next_index = 0
        
        self.is_trained = False
        self.vector_count = 0
    
    async def initialize(self) -> None:
        """Initialize the FAISS index."""
        try:
            def _create_index():
                import faiss
                
                if self.index_type == "Flat":
                    if self.metric == "L2":
                        return faiss.IndexFlatL2(self.dimension)
                    else:  # IP (Inner Product)
                        return faiss.IndexFlatIP(self.dimension)
                
                elif self.index_type == "IVF":
                    # IVF with 100 centroids (adjust based on data size)
                    nlist = min(100, max(1, self.vector_count // 100))
                    quantizer = faiss.IndexFlatL2(self.dimension)
                    if self.metric == "L2":
                        return faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                    else:
                        return faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                
                elif self.index_type == "HNSW":
                    # HNSW with M=16 connections
                    index = faiss.IndexHNSWFlat(self.dimension, 16)
                    if self.metric == "IP":
                        index.metric_type = faiss.METRIC_INNER_PRODUCT
                    return index
                
                else:
                    raise ValueError(f"Unsupported index type: {self.index_type}")
            
            self.index = await asyncio.get_event_loop().run_in_executor(None, _create_index)
            
            self.logger.info(
                "FAISS index initialized",
                index_type=self.index_type,
                dimension=self.dimension,
                metric=self.metric
            )
            
        except ImportError:
            raise VectorDatabaseError(
                "FAISS not installed. Install with: pip install faiss-cpu"
            )
        except Exception as e:
            raise VectorDatabaseError(f"Failed to initialize FAISS index: {str(e)}") from e
    
    async def add_vectors(
        self, 
        vectors: List[List[float]], 
        document_ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add vectors to the index.
        
        Args:
            vectors: List of embedding vectors
            document_ids: List of document IDs
            metadata: Optional metadata for each document
        """
        if not self.index:
            await self.initialize()
        
        if len(vectors) != len(document_ids):
            raise VectorDatabaseError("Number of vectors must match number of document IDs")
        
        try:
            def _add_vectors():
                import faiss
                
                # Convert to numpy array
                vectors_np = np.array(vectors, dtype=np.float32)
                
                # Normalize vectors if using inner product
                if self.metric == "IP":
                    faiss.normalize_L2(vectors_np)
                
                # Train index if needed (for IVF)
                if self.index_type == "IVF" and not self.is_trained:
                    if len(vectors) >= 100:  # Need enough vectors to train
                        self.index.train(vectors_np)
                        self.is_trained = True
                    else:
                        # Not enough vectors to train, use Flat index temporarily
                        self.logger.warning(
                            "Not enough vectors to train IVF index, using Flat index",
                            vector_count=len(vectors)
                        )
                        return
                
                # Add vectors to index
                start_idx = self.next_index
                self.index.add(vectors_np)
                
                # Update mappings
                for i, doc_id in enumerate(document_ids):
                    idx = start_idx + i
                    self.id_to_index[doc_id] = idx
                    self.index_to_id[idx] = doc_id
                    
                    # Store metadata
                    if metadata and i < len(metadata):
                        self.metadata_store[doc_id] = metadata[i]
                
                self.next_index += len(vectors)
                self.vector_count += len(vectors)
            
            await asyncio.get_event_loop().run_in_executor(None, _add_vectors)
            
            self.logger.info(
                "Vectors added to index",
                vector_count=len(vectors),
                total_vectors=self.vector_count
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to add vectors to index",
                vector_count=len(vectors),
                error=str(e)
            )
            raise IndexError(f"Failed to add vectors: {str(e)}") from e
    
    @log_performance("vector_search")
    async def search(
        self, 
        query_vector: List[float], 
        k: int = 10,
        threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results
        """
        if not self.index or self.vector_count == 0:
            return []
        
        try:
            def _search():
                import faiss
                
                # Convert query to numpy array
                query_np = np.array([query_vector], dtype=np.float32)
                
                # Normalize if using inner product
                if self.metric == "IP":
                    faiss.normalize_L2(query_np)
                
                # Perform search
                distances, indices = self.index.search(query_np, k)
                
                return distances[0], indices[0]
            
            distances, indices = await asyncio.get_event_loop().run_in_executor(None, _search)
            
            # Convert results
            results = []
            for i, (distance, idx) in enumerate(zip(distances, indices)):
                if idx == -1:  # FAISS returns -1 for invalid results
                    continue
                
                # Convert distance to similarity score
                if self.metric == "L2":
                    # For L2 distance, convert to similarity (0-1 range)
                    similarity = 1.0 / (1.0 + distance)
                else:  # IP
                    # For inner product, distance is already similarity
                    similarity = float(distance)
                
                # Apply threshold if specified
                if threshold and similarity < threshold:
                    continue
                
                # Get document ID and metadata
                doc_id = self.index_to_id.get(int(idx))
                if not doc_id:
                    continue
                
                metadata = self.metadata_store.get(doc_id, {})
                
                result = SearchResult(
                    document_id=doc_id,
                    score=similarity,
                    title=metadata.get('title'),
                    snippet=metadata.get('snippet'),
                    metadata=metadata
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(
                "Vector search failed",
                k=k,
                error=str(e)
            )
            raise SearchError(f"Vector search failed: {str(e)}") from e
    
    async def batch_search(
        self,
        query_vectors: List[List[float]],
        k: int = 10,
        threshold: Optional[float] = None
    ) -> List[List[SearchResult]]:
        """
        Perform batch search for multiple queries.
        
        Args:
            query_vectors: List of query vectors
            k: Number of results per query
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results for each query
        """
        if not self.index or self.vector_count == 0:
            return [[] for _ in query_vectors]
        
        try:
            def _batch_search():
                import faiss
                
                # Convert queries to numpy array
                queries_np = np.array(query_vectors, dtype=np.float32)
                
                # Normalize if using inner product
                if self.metric == "IP":
                    faiss.normalize_L2(queries_np)
                
                # Perform batch search
                distances, indices = self.index.search(queries_np, k)
                
                return distances, indices
            
            distances, indices = await asyncio.get_event_loop().run_in_executor(None, _batch_search)
            
            # Convert results for each query
            all_results = []
            for query_idx in range(len(query_vectors)):
                query_results = []
                
                for i in range(k):
                    idx = indices[query_idx][i]
                    distance = distances[query_idx][i]
                    
                    if idx == -1:
                        continue
                    
                    # Convert distance to similarity
                    if self.metric == "L2":
                        similarity = 1.0 / (1.0 + distance)
                    else:
                        similarity = float(distance)
                    
                    if threshold and similarity < threshold:
                        continue
                    
                    doc_id = self.index_to_id.get(int(idx))
                    if not doc_id:
                        continue
                    
                    metadata = self.metadata_store.get(doc_id, {})
                    
                    result = SearchResult(
                        document_id=doc_id,
                        score=similarity,
                        title=metadata.get('title'),
                        snippet=metadata.get('snippet'),
                        metadata=metadata
                    )
                    query_results.append(result)
                
                all_results.append(query_results)
            
            self.logger.info(
                "Batch search completed",
                query_count=len(query_vectors),
                k=k,
                total_results=sum(len(results) for results in all_results)
            )
            
            return all_results
            
        except Exception as e:
            self.logger.error(
                "Batch search failed",
                query_count=len(query_vectors),
                error=str(e)
            )
            raise SearchError(f"Batch search failed: {str(e)}") from e
    
    async def save_index(self, path: Optional[Path] = None) -> None:
        """Save the index and metadata to disk."""
        if not self.index:
            return
        
        save_path = path or self.storage_path
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            def _save():
                import faiss
                
                # Save FAISS index
                index_path = save_path / "faiss.index"
                faiss.write_index(self.index, str(index_path))
                
                # Save metadata
                metadata_path = save_path / "metadata.pkl"
                with open(metadata_path, 'wb') as f:
                    pickle.dump({
                        'metadata_store': self.metadata_store,
                        'id_to_index': self.id_to_index,
                        'index_to_id': self.index_to_id,
                        'next_index': self.next_index,
                        'vector_count': self.vector_count,
                        'dimension': self.dimension,
                        'index_type': self.index_type,
                        'metric': self.metric,
                        'is_trained': self.is_trained
                    }, f)
            
            await asyncio.get_event_loop().run_in_executor(None, _save)
            
            self.logger.info(
                "Index saved successfully",
                path=str(save_path),
                vector_count=self.vector_count
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to save index",
                path=str(save_path),
                error=str(e)
            )
            raise VectorDatabaseError(f"Failed to save index: {str(e)}") from e
    
    async def load_index(self, path: Optional[Path] = None) -> None:
        """Load the index and metadata from disk."""
        load_path = path or self.storage_path
        
        index_path = load_path / "faiss.index"
        metadata_path = load_path / "metadata.pkl"
        
        if not index_path.exists() or not metadata_path.exists():
            self.logger.warning(
                "Index files not found, starting with empty index",
                path=str(load_path)
            )
            await self.initialize()
            return
        
        try:
            def _load():
                import faiss
                
                # Load FAISS index
                index = faiss.read_index(str(index_path))
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                
                return index, data
            
            index, data = await asyncio.get_event_loop().run_in_executor(None, _load)
            
            # Restore state
            self.index = index
            self.metadata_store = data['metadata_store']
            self.id_to_index = data['id_to_index']
            self.index_to_id = data['index_to_id']
            self.next_index = data['next_index']
            self.vector_count = data['vector_count']
            self.dimension = data['dimension']
            self.index_type = data['index_type']
            self.metric = data['metric']
            self.is_trained = data['is_trained']
            
            self.logger.info(
                "Index loaded successfully",
                path=str(load_path),
                vector_count=self.vector_count
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to load index",
                path=str(load_path),
                error=str(e)
            )
            raise VectorDatabaseError(f"Failed to load index: {str(e)}") from e
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "vector_count": self.vector_count,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "is_trained": self.is_trained,
            "memory_usage": self.index.ntotal * self.dimension * 4 if self.index else 0,  # bytes
            "storage_path": str(self.storage_path)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the vector store."""
        try:
            if not self.index:
                await self.initialize()
            
            return {
                "status": "healthy",
                "stats": self.get_stats()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
