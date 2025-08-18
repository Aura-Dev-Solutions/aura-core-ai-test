"""
Embedding model implementation using sentence-transformers.
Provides high-quality embeddings for semantic search and similarity.
"""

import asyncio
from typing import List, Union, Optional, Dict, Any
from pathlib import Path
import numpy as np

from src.core.exceptions import ModelLoadError, EmbeddingGenerationError
from src.core.config import settings
from src.core.logging import log_performance
from .base import BaseEmbeddingModel


class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):
    """
    Embedding model using sentence-transformers.
    
    Supports various pre-trained models optimized for different use cases:
    - all-MiniLM-L6-v2: Fast and efficient, good for most use cases
    - all-mpnet-base-v2: Higher quality, slower
    - multi-qa-MiniLM-L6-cos-v1: Optimized for Q&A and retrieval
    """
    
    def __init__(
        self, 
        model_name: str = None,
        device: str = "cpu",
        normalize_embeddings: bool = True,
        max_seq_length: int = 512
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformer model
            device: Device to run the model on ('cpu' or 'cuda')
            normalize_embeddings: Whether to normalize embeddings
            max_seq_length: Maximum sequence length
        """
        model_name = model_name or settings.embedding_model
        super().__init__(model_name)
        
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.max_seq_length = max_seq_length
        self.embedding_dimension = settings.embedding_dimension
        
        # Model selection justification
        self.model_info = {
            "all-MiniLM-L6-v2": {
                "dimension": 384,
                "description": "Fast and efficient, good balance of speed and quality",
                "use_case": "General purpose, production ready",
                "speed": "fast",
                "quality": "good"
            },
            "all-mpnet-base-v2": {
                "dimension": 768,
                "description": "Higher quality embeddings, slower inference",
                "use_case": "High accuracy requirements",
                "speed": "medium",
                "quality": "excellent"
            },
            "multi-qa-MiniLM-L6-cos-v1": {
                "dimension": 384,
                "description": "Optimized for question-answering and retrieval",
                "use_case": "Q&A systems, document retrieval",
                "speed": "fast",
                "quality": "good"
            }
        }
    
    async def load_model(self) -> None:
        """Load the sentence-transformer model."""
        try:
            self.logger.info(
                "Loading embedding model",
                model_name=self.model_name,
                device=self.device
            )
            
            def _load_model():
                try:
                    from sentence_transformers import SentenceTransformer
                    
                    model = SentenceTransformer(
                        self.model_name,
                        device=self.device
                    )
                    
                    # Set max sequence length
                    if hasattr(model, 'max_seq_length'):
                        model.max_seq_length = self.max_seq_length
                    
                    return model
                    
                except ImportError:
                    raise ModelLoadError(
                        "sentence-transformers not installed. "
                        "Install with: pip install sentence-transformers"
                    )
                except Exception as e:
                    raise ModelLoadError(f"Failed to load model {self.model_name}: {str(e)}")
            
            # Load model in thread pool to avoid blocking
            self.model = await asyncio.get_event_loop().run_in_executor(None, _load_model)
            
            # Update embedding dimension from actual model
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            self.is_loaded = True
            
            self.logger.info(
                "Embedding model loaded successfully",
                model_name=self.model_name,
                dimension=self.embedding_dimension,
                device=self.device
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to load embedding model",
                model_name=self.model_name,
                error=str(e)
            )
            raise ModelLoadError(f"Failed to load embedding model: {str(e)}") from e
    
    async def unload_model(self) -> None:
        """Unload the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False
            
            self.logger.info(
                "Embedding model unloaded",
                model_name=self.model_name
            )
    
    @log_performance("embedding_generation")
    async def predict(self, input_data: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Make predictions with the model (alias for encode).
        
        Args:
            input_data: Text(s) to encode
            
        Returns:
            Embedding(s)
        """
        return await self.encode(input_data)
    
    async def encode(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            
        Returns:
            Single embedding or list of embeddings
        """
        await self.ensure_loaded()
        
        try:
            is_single = isinstance(texts, str)
            if is_single:
                texts = [texts]
            
            def _encode():
                embeddings = self.model.encode(
                    texts,
                    normalize_embeddings=self.normalize_embeddings,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                return embeddings.tolist()
            
            # Run encoding in thread pool
            embeddings = await asyncio.get_event_loop().run_in_executor(None, _encode)
            
            return embeddings[0] if is_single else embeddings
            
        except Exception as e:
            self.logger.error(
                "Embedding generation failed",
                model_name=self.model_name,
                text_count=len(texts) if isinstance(texts, list) else 1,
                error=str(e)
            )
            raise EmbeddingGenerationError(f"Failed to generate embeddings: {str(e)}") from e
    
    async def encode_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Encode texts in batches for better performance.
        
        Args:
            texts: List of texts to encode
            batch_size: Size of each batch
            show_progress: Whether to show progress bar
            
        Returns:
            List of embeddings
        """
        await self.ensure_loaded()
        
        if not texts:
            return []
        
        try:
            def _encode_batch():
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=self.normalize_embeddings,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True
                )
                return embeddings.tolist()
            
            embeddings = await asyncio.get_event_loop().run_in_executor(None, _encode_batch)
            
            self.logger.info(
                "Batch encoding completed",
                text_count=len(texts),
                batch_size=batch_size,
                embedding_dimension=len(embeddings[0]) if embeddings else 0
            )
            
            return embeddings
            
        except Exception as e:
            self.logger.error(
                "Batch embedding generation failed",
                text_count=len(texts),
                batch_size=batch_size,
                error=str(e)
            )
            raise EmbeddingGenerationError(f"Failed to generate batch embeddings: {str(e)}") from e
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.embedding_dimension
    
    async def similarity(
        self, 
        embeddings1: Union[List[float], List[List[float]]], 
        embeddings2: Union[List[float], List[List[float]]]
    ) -> Union[float, List[float], List[List[float]]]:
        """
        Calculate cosine similarity between embeddings.
        
        Args:
            embeddings1: First embedding(s)
            embeddings2: Second embedding(s)
            
        Returns:
            Similarity score(s)
        """
        try:
            def _calculate_similarity():
                from sklearn.metrics.pairwise import cosine_similarity
                
                # Convert to numpy arrays
                emb1 = np.array(embeddings1)
                emb2 = np.array(embeddings2)
                
                # Ensure 2D arrays
                if emb1.ndim == 1:
                    emb1 = emb1.reshape(1, -1)
                if emb2.ndim == 1:
                    emb2 = emb2.reshape(1, -1)
                
                similarity_matrix = cosine_similarity(emb1, emb2)
                
                # Return appropriate format
                if similarity_matrix.shape == (1, 1):
                    return float(similarity_matrix[0, 0])
                elif similarity_matrix.shape[0] == 1:
                    return similarity_matrix[0].tolist()
                elif similarity_matrix.shape[1] == 1:
                    return similarity_matrix[:, 0].tolist()
                else:
                    return similarity_matrix.tolist()
            
            return await asyncio.get_event_loop().run_in_executor(None, _calculate_similarity)
            
        except Exception as e:
            self.logger.error(
                "Similarity calculation failed",
                error=str(e)
            )
            raise EmbeddingGenerationError(f"Failed to calculate similarity: {str(e)}") from e
    
    async def find_most_similar(
        self, 
        query_embedding: List[float], 
        candidate_embeddings: List[List[float]], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find most similar embeddings to a query.
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of similarity results with indices and scores
        """
        try:
            similarities = await self.similarity(query_embedding, candidate_embeddings)
            
            # Create results with indices
            results = [
                {"index": i, "score": float(score)}
                for i, score in enumerate(similarities)
            ]
            
            # Sort by score (descending) and take top_k
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            self.logger.error(
                "Most similar search failed",
                query_dim=len(query_embedding),
                candidate_count=len(candidate_embeddings),
                error=str(e)
            )
            raise EmbeddingGenerationError(f"Failed to find most similar: {str(e)}") from e
    
    def get_model_justification(self) -> Dict[str, Any]:
        """
        Get justification for model selection.
        
        Returns:
            Dictionary with model selection reasoning
        """
        return {
            "selected_model": self.model_name,
            "justification": {
                "performance": "Optimized for production use with good speed/quality balance",
                "size": "Compact model suitable for deployment constraints",
                "multilingual": "Supports multiple languages if needed",
                "domain": "General purpose, works well for document analysis",
                "maintenance": "Actively maintained by sentence-transformers team"
            },
            "alternatives_considered": {
                "OpenAI embeddings": "Requires API calls, cost and latency concerns",
                "BERT-base": "Larger model, slower inference",
                "Universal Sentence Encoder": "TensorFlow dependency, less flexible"
            },
            "model_info": self.model_info.get(self.model_name, {}),
            "technical_specs": {
                "dimension": self.embedding_dimension,
                "max_sequence_length": self.max_seq_length,
                "normalization": self.normalize_embeddings,
                "device": self.device
            }
        }
