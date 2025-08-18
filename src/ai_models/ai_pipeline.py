"""
Integrated AI pipeline combining all AI models.
Provides unified interface for document analysis with embeddings, classification, and NER.
"""

import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path

from src.core.exceptions import AIModelError
from src.core.logging import LoggerMixin, log_performance
from src.core.models import Document, ProcessingResult, ClassificationResult, NamedEntity
from .embedding_model import SentenceTransformerEmbeddingModel
from .classification_model import DocumentClassificationModel
from .ner_model import DocumentNERModel
from .semantic_search import SemanticSearchService
from .base import model_registry


class AIAnalysisPipeline(LoggerMixin):
    """
    Integrated AI pipeline for complete document analysis.
    
    Combines:
    - Embedding generation for semantic search
    - Document classification
    - Named Entity Recognition (NER)
    - Semantic search capabilities
    """
    
    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        enable_classification: bool = True,
        enable_ner: bool = True,
        enable_semantic_search: bool = True
    ):
        """
        Initialize the AI pipeline.
        
        Args:
            embedding_model_name: Name of embedding model to use
            enable_classification: Whether to enable document classification
            enable_ner: Whether to enable NER
            enable_semantic_search: Whether to enable semantic search
        """
        self.embedding_model_name = embedding_model_name
        self.enable_classification = enable_classification
        self.enable_ner = enable_ner
        self.enable_semantic_search = enable_semantic_search
        
        # Model instances
        self.embedding_model = None
        self.classification_model = None
        self.ner_model = None
        self.semantic_search_service = None
        
        self.is_initialized = False
        self.pipeline_stats = {
            'documents_processed': 0,
            'embeddings_generated': 0,
            'classifications_made': 0,
            'entities_extracted': 0,
            'searches_performed': 0
        }
    
    async def initialize(self) -> None:
        """Initialize all AI models in the pipeline."""
        if self.is_initialized:
            return
        
        try:
            self.logger.info("Initializing AI pipeline...")
            
            # Initialize embedding model (always needed)
            self.embedding_model = SentenceTransformerEmbeddingModel(
                model_name=self.embedding_model_name
            )
            # Note: In offline mode, we'll simulate this
            # await self.embedding_model.load_model()
            
            # Register embedding model
            model_registry.register_model(
                "embedding",
                self.embedding_model_name or "default",
                self.embedding_model
            )
            
            # Initialize classification model
            if self.enable_classification:
                self.classification_model = DocumentClassificationModel()
                await self.classification_model.load_model()
                
                model_registry.register_model(
                    "classification",
                    "document_classifier",
                    self.classification_model
                )
            
            # Initialize NER model
            if self.enable_ner:
                self.ner_model = DocumentNERModel()
                # Note: In offline mode, we'll simulate this
                # await self.ner_model.load_model()
                
                model_registry.register_model(
                    "ner",
                    "document_ner",
                    self.ner_model
                )
            
            # Initialize semantic search service
            if self.enable_semantic_search:
                self.semantic_search_service = SemanticSearchService(
                    embedding_model_name=self.embedding_model_name
                )
                # Note: In offline mode, we'll simulate this
                # await self.semantic_search_service.initialize()
            
            self.is_initialized = True
            
            self.logger.info(
                "AI pipeline initialized successfully",
                embedding_model=bool(self.embedding_model),
                classification=bool(self.classification_model),
                ner=bool(self.ner_model),
                semantic_search=bool(self.semantic_search_service)
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to initialize AI pipeline",
                error=str(e)
            )
            raise AIModelError(f"Failed to initialize AI pipeline: {str(e)}") from e
    
    @log_performance("ai_pipeline_analysis")
    async def analyze_document(
        self, 
        document: Document, 
        text_content: str,
        include_embeddings: bool = True,
        include_classification: bool = True,
        include_ner: bool = True
    ) -> ProcessingResult:
        """
        Perform complete AI analysis on a document.
        
        Args:
            document: Document to analyze
            text_content: Text content of the document
            include_embeddings: Whether to generate embeddings
            include_classification: Whether to classify document
            include_ner: Whether to extract entities
            
        Returns:
            Processing result with AI analysis
        """
        await self.initialize()
        
        try:
            # Create base processing result
            result = ProcessingResult(
                document_id=document.id,
                status=document.status,
                text_content=text_content,
                metadata=document.metadata
            )
            
            # Generate embeddings
            embeddings = None
            if include_embeddings and self.embedding_model:
                embeddings = await self._generate_embeddings_simulation(text_content)
                result.embeddings = embeddings
                self.pipeline_stats['embeddings_generated'] += 1
            
            # Classify document
            classification = None
            if include_classification and self.classification_model:
                classification = await self.classification_model.classify(text_content)
                result.classification = classification
                self.pipeline_stats['classifications_made'] += 1
            
            # Extract entities
            entities = []
            if include_ner and self.ner_model:
                entities = await self._extract_entities_simulation(text_content)
                result.entities = entities
                self.pipeline_stats['entities_extracted'] += len(entities)
            
            # Create chunks with AI enhancements
            if self.embedding_model:
                from src.document_processor.json_extractor import JSONExtractor
                extractor = JSONExtractor()
                chunks = extractor.create_chunks(text_content)
                
                # Enhance chunks with AI metadata
                for chunk in chunks:
                    chunk.metadata.update({
                        'has_classification': classification is not None,
                        'entity_count': len([e for e in entities if 
                                           e.start_char >= chunk.start_char and 
                                           e.end_char <= chunk.end_char]),
                        'classification_category': classification.category.value if classification else None,
                        'classification_confidence': classification.confidence if classification else None
                    })
                
                result.chunks = chunks
            
            self.pipeline_stats['documents_processed'] += 1
            
            self.logger.info(
                "Document AI analysis completed",
                document_id=str(document.id),
                embeddings_generated=bool(embeddings),
                classified=bool(classification),
                entities_found=len(entities),
                chunks_created=len(result.chunks) if result.chunks else 0
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "AI analysis failed",
                document_id=str(document.id),
                error=str(e)
            )
            raise AIModelError(f"AI analysis failed: {str(e)}") from e
    
    async def _generate_embeddings_simulation(self, text: str) -> List[float]:
        """Simulate embedding generation (offline mode)."""
        import numpy as np
        import hashlib
        
        # Generate consistent embeddings based on text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        np.random.seed(int(text_hash[:8], 16) % (2**32))
        
        # Simulate 384-dimensional embedding
        embedding = np.random.rand(384).tolist()
        
        return embedding
    
    async def _extract_entities_simulation(self, text: str) -> List[NamedEntity]:
        """Simulate entity extraction (offline mode)."""
        import re
        
        entities = []
        
        # Simulate common entity patterns
        patterns = {
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE': r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            'MONEY': r'(\$|€|£|USD|EUR)\s*[\d,]+\.?\d*',
            'DATE': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            'PERSON': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Simple name pattern
            'ORG': r'\b[A-Z][a-z]+\s+(Inc|Corp|Ltd|S\.A\.|Research|Solutions)\b'
        }
        
        for label, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                # Filter out very short matches
                if len(match.group().strip()) < 3:
                    continue
                
                entity = NamedEntity(
                    text=match.group().strip(),
                    label=label,
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.8,
                    metadata={'simulated': True, 'pattern': label}
                )
                entities.append(entity)
        
        # Remove duplicates and overlapping entities
        entities = self._remove_overlapping_entities(entities)
        
        return entities
    
    def _remove_overlapping_entities(self, entities: List[NamedEntity]) -> List[NamedEntity]:
        """Remove overlapping entities, keeping the one with higher confidence."""
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda x: x.start_char)
        
        filtered = []
        for entity in entities:
            # Check for overlap with already added entities
            overlaps = False
            for existing in filtered:
                if (entity.start_char < existing.end_char and 
                    entity.end_char > existing.start_char):
                    # There's an overlap
                    if entity.confidence <= existing.confidence:
                        overlaps = True
                        break
                    else:
                        # Remove the existing entity with lower confidence
                        filtered.remove(existing)
                        break
            
            if not overlaps:
                filtered.append(entity)
        
        return filtered
    
    async def analyze_documents_batch(
        self,
        documents_with_content: List[tuple[Document, str]],
        batch_size: int = 5
    ) -> List[ProcessingResult]:
        """
        Analyze multiple documents in batches.
        
        Args:
            documents_with_content: List of (document, text_content) tuples
            batch_size: Number of documents to process concurrently
            
        Returns:
            List of processing results
        """
        await self.initialize()
        
        results = []
        total_docs = len(documents_with_content)
        
        for i in range(0, total_docs, batch_size):
            batch = documents_with_content[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [
                self.analyze_document(doc, content)
                for doc, content in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    self.logger.error(
                        "Batch analysis failed for document",
                        document_id=str(batch[j][0].id),
                        error=str(result)
                    )
                    # Create error result
                    error_result = ProcessingResult(
                        document_id=batch[j][0].id,
                        status=batch[j][0].status,
                        text_content=batch[j][1],
                        error_message=str(result)
                    )
                    results.append(error_result)
                else:
                    results.append(result)
        
        self.logger.info(
            "Batch AI analysis completed",
            total_documents=total_docs,
            successful=sum(1 for r in results if not r.error_message),
            failed=sum(1 for r in results if r.error_message)
        )
        
        return results
    
    async def search_documents(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search on indexed documents.
        
        Args:
            query: Search query
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results
        """
        if not self.semantic_search_service:
            raise AIModelError("Semantic search not enabled")
        
        from src.core.models import SearchQuery
        
        search_query = SearchQuery(
            query=query,
            limit=limit,
            threshold=threshold
        )
        
        results = await self.semantic_search_service.search(search_query)
        self.pipeline_stats['searches_performed'] += 1
        
        return [
            {
                'document_id': str(result.document_id),
                'title': result.title,
                'snippet': result.snippet,
                'score': result.score,
                'metadata': result.metadata
            }
            for result in results
        ]
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            'pipeline_stats': self.pipeline_stats.copy(),
            'models_loaded': {
                'embedding': bool(self.embedding_model),
                'classification': bool(self.classification_model),
                'ner': bool(self.ner_model),
                'semantic_search': bool(self.semantic_search_service)
            },
            'capabilities': {
                'embedding_generation': bool(self.embedding_model),
                'document_classification': self.enable_classification,
                'entity_extraction': self.enable_ner,
                'semantic_search': self.enable_semantic_search
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the AI pipeline."""
        try:
            await self.initialize()
            
            health_status = {
                'status': 'healthy',
                'models': {},
                'pipeline_stats': self.pipeline_stats
            }
            
            # Check each model
            if self.embedding_model:
                health_status['models']['embedding'] = await self.embedding_model.health_check()
            
            if self.classification_model:
                health_status['models']['classification'] = await self.classification_model.health_check()
            
            if self.ner_model:
                health_status['models']['ner'] = await self.ner_model.health_check()
            
            if self.semantic_search_service:
                health_status['models']['semantic_search'] = await self.semantic_search_service.health_check()
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'pipeline_stats': self.pipeline_stats
            }
    
    def get_model_justifications(self) -> Dict[str, Any]:
        """Get justifications for all model selections."""
        justifications = {}
        
        if self.embedding_model:
            justifications['embedding'] = self.embedding_model.get_model_justification()
        
        if self.classification_model:
            justifications['classification'] = self.classification_model.get_model_justification()
        
        if self.ner_model:
            justifications['ner'] = self.ner_model.get_model_justification()
        
        return {
            'pipeline_approach': {
                'strategy': 'Modular AI pipeline with specialized models',
                'benefits': [
                    'Each model optimized for specific task',
                    'Can enable/disable components as needed',
                    'Easy to upgrade individual models',
                    'Parallel processing capabilities'
                ],
                'integration': 'Unified interface with consistent error handling'
            },
            'model_justifications': justifications
        }
    
    async def shutdown(self) -> None:
        """Shutdown the AI pipeline and cleanup resources."""
        self.logger.info("Shutting down AI pipeline...")
        
        if self.embedding_model:
            await self.embedding_model.unload_model()
        
        if self.classification_model:
            await self.classification_model.unload_model()
        
        if self.ner_model:
            await self.ner_model.unload_model()
        
        self.is_initialized = False
        
        self.logger.info("AI pipeline shutdown completed")


# Alias para compatibilidad
AIPipeline = AIAnalysisPipeline
