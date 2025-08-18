"""
Base classes for AI/ML models in Aura Document Analyzer.
Provides common interface and functionality for all AI models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import asyncio
from pathlib import Path

from src.core.exceptions import AIModelError, ModelLoadError, ModelNotFoundError
from src.core.logging import LoggerMixin, log_performance
from src.core.config import settings


class BaseAIModel(ABC, LoggerMixin):
    """Abstract base class for all AI models."""
    
    def __init__(self, model_name: str, model_path: Optional[Path] = None):
        """
        Initialize the AI model.
        
        Args:
            model_name: Name/identifier of the model
            model_path: Optional path to model files
        """
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.model_info = {}
    
    @abstractmethod
    async def load_model(self) -> None:
        """Load the model into memory."""
        pass
    
    @abstractmethod
    async def unload_model(self) -> None:
        """Unload the model from memory."""
        pass
    
    @abstractmethod
    async def predict(self, input_data: Any) -> Any:
        """Make predictions with the model."""
        pass
    
    async def ensure_loaded(self) -> None:
        """Ensure the model is loaded before use."""
        if not self.is_loaded:
            await self.load_model()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "name": self.model_name,
            "path": str(self.model_path) if self.model_path else None,
            "loaded": self.is_loaded,
            "info": self.model_info
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the model."""
        try:
            await self.ensure_loaded()
            return {
                "status": "healthy",
                "model": self.model_name,
                "loaded": self.is_loaded
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "model": self.model_name,
                "error": str(e)
            }


class BaseEmbeddingModel(BaseAIModel):
    """Base class for embedding models."""
    
    @abstractmethod
    async def encode(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            
        Returns:
            Single embedding or list of embeddings
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        pass
    
    async def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Encode texts in batches for better performance.
        
        Args:
            texts: List of texts to encode
            batch_size: Size of each batch
            
        Returns:
            List of embeddings
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await self.encode(batch)
            
            if isinstance(batch_embeddings[0], list):
                embeddings.extend(batch_embeddings)
            else:
                embeddings.append(batch_embeddings)
        
        return embeddings


class BaseClassificationModel(BaseAIModel):
    """Base class for classification models."""
    
    @abstractmethod
    async def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify a text.
        
        Args:
            text: Text to classify
            
        Returns:
            Classification result with probabilities
        """
        pass
    
    @abstractmethod
    def get_classes(self) -> List[str]:
        """Get list of possible classes."""
        pass


class BaseNERModel(BaseAIModel):
    """Base class for Named Entity Recognition models."""
    
    @abstractmethod
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        Args:
            text: Text to process
            
        Returns:
            List of entities with positions and labels
        """
        pass
    
    @abstractmethod
    def get_entity_types(self) -> List[str]:
        """Get list of entity types this model can recognize."""
        pass


class ModelRegistry:
    """Registry for managing AI models."""
    
    def __init__(self):
        self._models: Dict[str, BaseAIModel] = {}
        self._embedding_models: Dict[str, BaseEmbeddingModel] = {}
        self._classification_models: Dict[str, BaseClassificationModel] = {}
        self._ner_models: Dict[str, BaseNERModel] = {}
    
    def register_model(self, model_type: str, model_name: str, model: BaseAIModel):
        """Register a model in the registry."""
        self._models[f"{model_type}:{model_name}"] = model
        
        if isinstance(model, BaseEmbeddingModel):
            self._embedding_models[model_name] = model
        elif isinstance(model, BaseClassificationModel):
            self._classification_models[model_name] = model
        elif isinstance(model, BaseNERModel):
            self._ner_models[model_name] = model
    
    def get_model(self, model_type: str, model_name: str) -> Optional[BaseAIModel]:
        """Get a model from the registry."""
        return self._models.get(f"{model_type}:{model_name}")
    
    def get_embedding_model(self, model_name: str) -> Optional[BaseEmbeddingModel]:
        """Get an embedding model."""
        return self._embedding_models.get(model_name)
    
    def get_classification_model(self, model_name: str) -> Optional[BaseClassificationModel]:
        """Get a classification model."""
        return self._classification_models.get(model_name)
    
    def get_ner_model(self, model_name: str) -> Optional[BaseNERModel]:
        """Get a NER model."""
        return self._ner_models.get(model_name)
    
    def list_models(self) -> Dict[str, List[str]]:
        """List all registered models by type."""
        return {
            "embedding": list(self._embedding_models.keys()),
            "classification": list(self._classification_models.keys()),
            "ner": list(self._ner_models.keys())
        }
    
    async def load_all_models(self):
        """Load all registered models."""
        for model in self._models.values():
            if not model.is_loaded:
                await model.load_model()
    
    async def unload_all_models(self):
        """Unload all models."""
        for model in self._models.values():
            if model.is_loaded:
                await model.unload_model()
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Health check for all models."""
        results = {}
        for key, model in self._models.items():
            results[key] = await model.health_check()
        return results


# Global model registry
model_registry = ModelRegistry()
