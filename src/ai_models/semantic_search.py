"""
Semantic search service that combines embedding generation and vector search.
Provides high-level interface for document indexing and semantic search.
"""

import asyncio
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from uuid import uuid4

from src.core.exceptions import SearchError, EmbeddingGenerationError
from src.core.logging import LoggerMixin, log_performance
from src.core.models import Document, SearchQuery, SearchResult, TextChunk
from src.core.config import settings
from .embedding_model import SentenceTransformerEmbeddingModel
from .vector_search import FAISSVectorStore
from .base import model_registry


class SemanticSearchService(LoggerMixin):
    """
    Semantic search service combining embeddings and vector search.
    
    Provides document indexing, semantic search, and similarity analysis.
    """
    
    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        vector_store_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the semantic search service.
        
        Args:
            embedding_model_name: Name of the embedding model to use
            vector_store_config: Configuration for vector store
        """
        self.embedding_model_name = embedding_model_name or settings.embedding_model
        self.vector_store_config = vector_store_config or {}
        
        self.embedding_model = None
        self.vector_store = None
        self.is_initialized = False
        
        # Document storage for metadata
        self.document_store: Dict[str, Document] = {}
        self.chunk_store: Dict[str, TextChunk] = {}
    
    async def initialize(self) -> None:
        """Initialize the embedding model and vector store."""
        if self.is_initialized:
            return
        
        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformerEmbeddingModel(
                model_name=self.embedding_model_name
            )
            await self.embedding_model.load_model()
            
            # Register model
            model_registry.register_model(
                "embedding", 
                self.embedding_model_name, 
                self.embedding_model
            )
            
            # Initialize vector store
            dimension = self.embedding_model.get_embedding_dimension()
            self.vector_store = FAISSVectorStore(
                dimension=dimension,
                **self.vector_store_config
            )
            await self.vector_store.initialize()
            
            # Try to load existing index
            await self.vector_store.load_index()
            
            self.is_initialized = True
            
            self.logger.info(
                "Semantic search service initialized",
                embedding_model=self.embedding_model_name,
                embedding_dimension=dimension,
                vector_store_type=self.vector_store.index_type
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to initialize semantic search service",
                error=str(e)
            )
            raise SearchError(f"Failed to initialize semantic search: {str(e)}") from e
    
    @log_performance("document_indexing")
    async def index_document(self, document: Document, text_content: str) -> None:
        """
        Index a document for semantic search.
        
        Args:
            document: Document to index
            text_content: Text content of the document
        """
        await self.initialize()
        
        try:
            # Store document
            self.document_store[str(document.id)] = document
            
            # Create chunks from text content
            from src.document_processor.base import BaseDocumentExtractor
            extractor = BaseDocumentExtractor()
            chunks = extractor.create_chunks(text_content)
            
            if not chunks:
                self.logger.warning(
                    "No chunks created for document",
                    document_id=str(document.id)
                )
                return
            
            # Generate embeddings for chunks
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = await self.embedding_model.encode_batch(chunk_texts)
            
            # Prepare data for vector store
            chunk_ids = []
            metadata_list = []
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk_id = f"{document.id}:{chunk.id}"
                chunk_ids.append(chunk_id)
                
                # Store chunk
                self.chunk_store[chunk_id] = chunk
                
                # Prepare metadata
                metadata = {
                    'document_id': str(document.id),
                    'chunk_id': chunk.id,
                    'title': document.metadata.title or document.filename,
                    'snippet': chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                    'page_number': chunk.page_number,
                    'section': chunk.section,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char,
                    'document_type': document.document_type,
                    'filename': document.filename
                }
                metadata_list.append(metadata)
            
            # Add to vector store
            await self.vector_store.add_vectors(embeddings, chunk_ids, metadata_list)
            
            # Save index
            await self.vector_store.save_index()
            
            self.logger.info(
                "Document indexed successfully",
                document_id=str(document.id),
                chunk_count=len(chunks),
                filename=document.filename
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to index document",
                document_id=str(document.id),
                error=str(e)
            )
            raise SearchError(f"Failed to index document: {str(e)}") from e
    
    @log_performance("batch_document_indexing")
    async def index_documents_batch(
        self, 
        documents_with_content: List[tuple[Document, str]],
        batch_size: int = 10
    ) -> None:
        """
        Index multiple documents in batches.
        
        Args:
            documents_with_content: List of (document, text_content) tuples
            batch_size: Number of documents to process in each batch
        """
        await self.initialize()
        
        total_docs = len(documents_with_content)
        successful = 0
        failed = 0
        
        for i in range(0, total_docs, batch_size):
            batch = documents_with_content[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [
                self.index_document(doc, content) 
                for doc, content in batch
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count results
            for result in results:
                if isinstance(result, Exception):
                    failed += 1
                else:
                    successful += 1
        
        self.logger.info(
            "Batch indexing completed",
            total_documents=total_docs,
            successful=successful,
            failed=failed
        )
    
    @log_performance("semantic_search")
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Perform semantic search.
        
        Args:
            query: Search query with parameters
            
        Returns:
            List of search results
        """
        await self.initialize()
        
        try:
            # Generate embedding for query
            query_embedding = await self.embedding_model.encode(query.query)
            
            # Perform vector search
            results = await self.vector_store.search(
                query_vector=query_embedding,
                k=query.limit,
                threshold=query.threshold
            )
            
            # Apply filters if specified
            if query.filters:
                results = self._apply_filters(results, query.filters)
            
            # Enhance results with additional metadata if requested
            if query.include_metadata:
                results = await self._enhance_results(results)
            
            self.logger.info(
                "Semantic search completed",
                query=query.query[:100],
                results_found=len(results),
                threshold=query.threshold
            )
            
            return results
            
        except Exception as e:
            self.logger.error(
                "Semantic search failed",
                query=query.query[:100],
                error=str(e)
            )
            raise SearchError(f"Semantic search failed: {str(e)}") from e
    
    async def find_similar_documents(
        self, 
        document_id: str, 
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[SearchResult]:
        """
        Find documents similar to a given document.
        
        Args:
            document_id: ID of the reference document
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar documents
        """
        await self.initialize()
        
        try:
            # Get document chunks
            document_chunks = [
                chunk_id for chunk_id in self.chunk_store.keys()
                if chunk_id.startswith(f"{document_id}:")
            ]
            
            if not document_chunks:
                return []
            
            # Get embeddings for document chunks
            chunk_embeddings = []
            for chunk_id in document_chunks:
                # This is a simplified approach - in practice, you'd store embeddings
                chunk = self.chunk_store[chunk_id]
                embedding = await self.embedding_model.encode(chunk.text)
                chunk_embeddings.append(embedding)
            
            # Average embeddings to represent document
            import numpy as np
            doc_embedding = np.mean(chunk_embeddings, axis=0).tolist()
            
            # Search for similar documents
            results = await self.vector_store.search(
                query_vector=doc_embedding,
                k=limit * 2,  # Get more results to filter out same document
                threshold=threshold
            )
            
            # Filter out chunks from the same document
            filtered_results = []
            seen_documents = set()
            
            for result in results:
                result_doc_id = result.metadata.get('document_id')
                if result_doc_id != document_id and result_doc_id not in seen_documents:
                    seen_documents.add(result_doc_id)
                    filtered_results.append(result)
                    
                    if len(filtered_results) >= limit:
                        break
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(
                "Similar document search failed",
                document_id=document_id,
                error=str(e)
            )
            raise SearchError(f"Similar document search failed: {str(e)}") from e
    
    async def get_document_summary(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get summary information about an indexed document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Document summary or None if not found
        """
        document = self.document_store.get(document_id)
        if not document:
            return None
        
        # Count chunks for this document
        chunk_count = sum(
            1 for chunk_id in self.chunk_store.keys()
            if chunk_id.startswith(f"{document_id}:")
        )
        
        return {
            'document_id': document_id,
            'filename': document.filename,
            'document_type': document.document_type,
            'title': document.metadata.title,
            'author': document.metadata.author,
            'chunk_count': chunk_count,
            'word_count': document.metadata.word_count,
            'created_at': document.created_at,
            'indexed': chunk_count > 0
        }
    
    def _apply_filters(self, results: List[SearchResult], filters: Dict[str, Any]) -> List[SearchResult]:
        """Apply filters to search results."""
        filtered_results = []
        
        for result in results:
            include = True
            
            for filter_key, filter_value in filters.items():
                metadata_value = result.metadata.get(filter_key)
                
                if filter_key == 'document_type' and metadata_value != filter_value:
                    include = False
                    break
                elif filter_key == 'min_score' and result.score < filter_value:
                    include = False
                    break
                elif filter_key == 'filename_contains' and filter_value.lower() not in result.metadata.get('filename', '').lower():
                    include = False
                    break
            
            if include:
                filtered_results.append(result)
        
        return filtered_results
    
    async def _enhance_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Enhance results with additional metadata."""
        for result in results:
            chunk_id = result.metadata.get('chunk_id')
            if chunk_id:
                full_chunk_id = f"{result.document_id}:{chunk_id}"
                chunk = self.chunk_store.get(full_chunk_id)
                if chunk:
                    result.metadata.update({
                        'chunk_text': chunk.text,
                        'chunk_start': chunk.start_char,
                        'chunk_end': chunk.end_char,
                        'chunk_metadata': chunk.metadata
                    })
        
        return results
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the semantic search service."""
        await self.initialize()
        
        vector_stats = self.vector_store.get_stats()
        
        return {
            'embedding_model': self.embedding_model_name,
            'embedding_dimension': self.embedding_model.get_embedding_dimension(),
            'indexed_documents': len(self.document_store),
            'indexed_chunks': len(self.chunk_store),
            'vector_store': vector_stats,
            'is_initialized': self.is_initialized
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the semantic search service."""
        try:
            await self.initialize()
            
            # Test embedding generation
            test_embedding = await self.embedding_model.encode("test query")
            
            # Test vector store
            vector_health = await self.vector_store.health_check()
            
            return {
                'status': 'healthy',
                'embedding_model_loaded': self.embedding_model.is_loaded,
                'embedding_dimension': len(test_embedding),
                'vector_store': vector_health,
                'stats': await self.get_stats()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
