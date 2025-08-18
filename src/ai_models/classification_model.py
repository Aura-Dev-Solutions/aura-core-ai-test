"""
Document classification model using machine learning.
Classifies documents into predefined categories with confidence scores.
"""

import asyncio
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from src.core.exceptions import ClassificationError, ModelLoadError
from src.core.config import settings
from src.core.logging import log_performance
from src.core.models import DocumentCategory, ClassificationResult
from .base import BaseClassificationModel


class DocumentClassificationModel(BaseClassificationModel):
    """
    Document classification model using TF-IDF + SVM.
    
    Classifies documents into categories:
    - CONTRACT: Legal contracts and agreements
    - REPORT: Technical reports and analysis
    - LEGAL: Legal documents and policies
    - CORRESPONDENCE: Business correspondence
    - TECHNICAL: Technical documentation
    - FINANCIAL: Financial documents
    - OTHER: Uncategorized documents
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the classification model.
        
        Args:
            model_path: Path to saved model files
        """
        super().__init__("document_classifier", model_path)
        
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = None
        
        # Category mapping and keywords
        self.category_keywords = {
            DocumentCategory.CONTRACT: [
                'contrato', 'contract', 'acuerdo', 'agreement', 'clausula', 'clause',
                'contratante', 'contratista', 'partes', 'parties', 'obligaciones',
                'terminos', 'terms', 'condiciones', 'conditions', 'firma', 'signature'
            ],
            DocumentCategory.REPORT: [
                'reporte', 'report', 'analisis', 'analysis', 'resultados', 'results',
                'metodologia', 'methodology', 'conclusiones', 'conclusions', 'datos',
                'data', 'estadisticas', 'statistics', 'metricas', 'metrics'
            ],
            DocumentCategory.LEGAL: [
                'legal', 'ley', 'law', 'politica', 'policy', 'privacidad', 'privacy',
                'proteccion', 'protection', 'derechos', 'rights', 'normativa',
                'regulation', 'cumplimiento', 'compliance', 'juridico'
            ],
            DocumentCategory.CORRESPONDENCE: [
                'estimado', 'dear', 'saludo', 'greeting', 'atentamente', 'sincerely',
                'propuesta', 'proposal', 'reunion', 'meeting', 'contacto', 'contact',
                'seguimiento', 'follow-up', 'comercial', 'business'
            ],
            DocumentCategory.TECHNICAL: [
                'manual', 'documentation', 'instalacion', 'installation', 'configuracion',
                'configuration', 'usuario', 'user', 'sistema', 'system', 'software',
                'hardware', 'tecnico', 'technical', 'especificaciones', 'specifications'
            ],
            DocumentCategory.FINANCIAL: [
                'financiero', 'financial', 'presupuesto', 'budget', 'costo', 'cost',
                'precio', 'price', 'factura', 'invoice', 'pago', 'payment',
                'inversion', 'investment', 'ganancia', 'profit', 'perdida', 'loss'
            ]
        }
    
    async def load_model(self) -> None:
        """Load or train the classification model."""
        try:
            if self.model_path and (self.model_path / "classifier.pkl").exists():
                await self._load_saved_model()
            else:
                await self._train_model()
            
            self.is_loaded = True
            self.logger.info(
                "Classification model loaded successfully",
                model_name=self.model_name,
                classes=len(self.get_classes())
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to load classification model",
                error=str(e)
            )
            raise ModelLoadError(f"Failed to load classification model: {str(e)}") from e
    
    async def _load_saved_model(self) -> None:
        """Load saved model from disk."""
        def _load():
            with open(self.model_path / "classifier.pkl", 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data['vectorizer']
            self.classifier = model_data['classifier']
            self.label_encoder = model_data['label_encoder']
            
            return model_data
        
        model_data = await asyncio.get_event_loop().run_in_executor(None, _load)
        self.model_info = model_data.get('info', {})
    
    async def _train_model(self) -> None:
        """Train the classification model with synthetic data."""
        def _train():
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.svm import SVC
            from sklearn.preprocessing import LabelEncoder
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import cross_val_score
            
            # Generate synthetic training data
            training_data = self._generate_training_data()
            
            texts = [item['text'] for item in training_data]
            labels = [item['category'] for item in training_data]
            
            # Create and train vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english',
                lowercase=True
            )
            
            # Create and train classifier
            self.classifier = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
            
            # Create label encoder
            self.label_encoder = LabelEncoder()
            encoded_labels = self.label_encoder.fit_transform(labels)
            
            # Fit vectorizer and classifier
            X = self.vectorizer.fit_transform(texts)
            self.classifier.fit(X, encoded_labels)
            
            # Calculate cross-validation score
            cv_scores = cross_val_score(self.classifier, X, encoded_labels, cv=5)
            
            self.model_info = {
                'training_samples': len(training_data),
                'features': X.shape[1],
                'classes': len(self.label_encoder.classes_),
                'cv_accuracy': float(np.mean(cv_scores)),
                'cv_std': float(np.std(cv_scores))
            }
            
            return self.model_info
        
        await asyncio.get_event_loop().run_in_executor(None, _train)
    
    def _generate_training_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic training data for each category."""
        training_data = []
        
        # Contract examples
        contract_texts = [
            "Este contrato de servicios establece los términos y condiciones entre las partes contratantes para el desarrollo de software.",
            "Las clausulas del presente acuerdo definen las obligaciones del contratista y los derechos del contratante.",
            "El contrato tiene una duración de seis meses con posibilidad de renovación según los términos establecidos.",
            "Service agreement between contractor and client for software development with specific terms and conditions.",
            "This contract establishes the legal framework for the provision of consulting services."
        ]
        
        # Report examples
        report_texts = [
            "Este reporte analiza el rendimiento de los modelos de inteligencia artificial utilizando métricas de precisión y recall.",
            "Los resultados del análisis muestran una mejora significativa en la eficiencia del sistema implementado.",
            "La metodología empleada incluye pruebas estadísticas y análisis comparativo de diferentes algoritmos.",
            "Performance analysis report showing significant improvements in system efficiency and user satisfaction metrics.",
            "Technical report on machine learning model evaluation with detailed statistical analysis and conclusions."
        ]
        
        # Legal examples
        legal_texts = [
            "Esta política de privacidad describe cómo recopilamos, utilizamos y protegemos la información personal de los usuarios.",
            "Los derechos de los usuarios incluyen el acceso, modificación y eliminación de sus datos personales según la normativa vigente.",
            "El cumplimiento de la ley de protección de datos es fundamental para nuestras operaciones comerciales.",
            "Privacy policy outlining data collection practices and user rights in accordance with GDPR regulations.",
            "Legal compliance document detailing regulatory requirements and organizational obligations."
        ]
        
        # Correspondence examples
        correspondence_texts = [
            "Estimado Sr. García, me dirijo a usted para hacer seguimiento a nuestra propuesta comercial enviada la semana pasada.",
            "Esperamos poder coordinar una reunión para discutir los detalles de la colaboración propuesta.",
            "Quedamos a la espera de su respuesta para proceder con los siguientes pasos del proceso comercial.",
            "Dear Mr. Johnson, I am writing to follow up on our business proposal submitted last week.",
            "Thank you for your interest in our services. We look forward to scheduling a meeting to discuss further."
        ]
        
        # Technical examples
        technical_texts = [
            "Manual de usuario del sistema de análisis de documentos con instrucciones de instalación y configuración.",
            "Los requisitos técnicos incluyen Python 3.11, 8GB de RAM y conexión a internet para la descarga de modelos.",
            "La documentación técnica describe la arquitectura del sistema y los procedimientos de mantenimiento.",
            "User manual for document analysis system including installation requirements and configuration steps.",
            "Technical specifications document outlining system architecture and deployment procedures."
        ]
        
        # Financial examples
        financial_texts = [
            "El presupuesto total del proyecto asciende a $50,000 USD distribuidos en seis meses de desarrollo.",
            "Los costos operativos incluyen licencias de software, infraestructura cloud y recursos humanos especializados.",
            "El análisis financiero muestra un retorno de inversión positivo en el primer año de implementación.",
            "Budget allocation for software development project with detailed cost breakdown and timeline.",
            "Financial analysis report showing projected costs, revenues, and return on investment calculations."
        ]
        
        # Add training examples
        categories_data = [
            (DocumentCategory.CONTRACT, contract_texts),
            (DocumentCategory.REPORT, report_texts),
            (DocumentCategory.LEGAL, legal_texts),
            (DocumentCategory.CORRESPONDENCE, correspondence_texts),
            (DocumentCategory.TECHNICAL, technical_texts),
            (DocumentCategory.FINANCIAL, financial_texts)
        ]
        
        for category, texts in categories_data:
            for text in texts:
                training_data.append({
                    'text': text,
                    'category': category.value
                })
        
        return training_data
    
    @log_performance("document_classification")
    async def classify(self, text: str) -> ClassificationResult:
        """
        Classify a document text.
        
        Args:
            text: Document text to classify
            
        Returns:
            Classification result with category and confidence
        """
        await self.ensure_loaded()
        
        try:
            def _classify():
                # Vectorize text
                X = self.vectorizer.transform([text])
                
                # Get prediction and probabilities
                prediction = self.classifier.predict(X)[0]
                probabilities = self.classifier.predict_proba(X)[0]
                
                # Decode prediction
                category_name = self.label_encoder.inverse_transform([prediction])[0]
                category = DocumentCategory(category_name)
                
                # Get confidence score
                confidence = float(max(probabilities))
                
                # Create probability dictionary
                prob_dict = {}
                for i, prob in enumerate(probabilities):
                    class_name = self.label_encoder.inverse_transform([i])[0]
                    prob_dict[class_name] = float(prob)
                
                return category, confidence, prob_dict
            
            category, confidence, probabilities = await asyncio.get_event_loop().run_in_executor(None, _classify)
            
            # Apply keyword-based boost
            keyword_boost = self._calculate_keyword_boost(text, category)
            final_confidence = min(1.0, confidence + keyword_boost)
            
            result = ClassificationResult(
                category=category,
                confidence=final_confidence,
                probabilities=probabilities,
                metadata={
                    'model_confidence': confidence,
                    'keyword_boost': keyword_boost,
                    'text_length': len(text),
                    'model_info': self.model_info
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Classification failed",
                text_length=len(text),
                error=str(e)
            )
            raise ClassificationError(f"Failed to classify document: {str(e)}") from e
    
    def _calculate_keyword_boost(self, text: str, predicted_category: DocumentCategory) -> float:
        """Calculate confidence boost based on keyword matching."""
        text_lower = text.lower()
        
        # Count keyword matches for predicted category
        category_keywords = self.category_keywords.get(predicted_category, [])
        matches = sum(1 for keyword in category_keywords if keyword in text_lower)
        
        # Calculate boost (max 0.1)
        boost = min(0.1, matches * 0.02)
        
        return boost
    
    async def classify_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple documents.
        
        Args:
            texts: List of document texts
            
        Returns:
            List of classification results
        """
        results = []
        for text in texts:
            result = await self.classify(text)
            results.append(result)
        
        return results
    
    def get_classes(self) -> List[str]:
        """Get list of possible classes."""
        return [category.value for category in DocumentCategory]
    
    async def save_model(self, path: Optional[Path] = None) -> None:
        """Save the trained model to disk."""
        if not self.is_loaded:
            return
        
        save_path = path or self.model_path or Path("data/models")
        save_path.mkdir(parents=True, exist_ok=True)
        
        def _save():
            model_data = {
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'label_encoder': self.label_encoder,
                'info': self.model_info,
                'category_keywords': self.category_keywords
            }
            
            with open(save_path / "classifier.pkl", 'wb') as f:
                pickle.dump(model_data, f)
        
        await asyncio.get_event_loop().run_in_executor(None, _save)
        
        self.logger.info(
            "Classification model saved",
            path=str(save_path)
        )
    
    async def predict(self, input_data: str) -> Dict[str, Any]:
        """
        Make predictions with the model (implements abstract method).
        
        Args:
            input_data: Text to classify
            
        Returns:
            Classification result as dictionary
        """
        result = await self.classify(input_data)
        return {
            'category': result.category.value,
            'confidence': result.confidence,
            'probabilities': result.probabilities
        }
    
    async def unload_model(self) -> None:
        """Unload the model from memory."""
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = None
        self.is_loaded = False
        
        self.logger.info("Classification model unloaded")
    
    def get_model_justification(self) -> Dict[str, Any]:
        """Get justification for model selection."""
        return {
            "selected_approach": "TF-IDF + SVM",
            "justification": {
                "interpretability": "SVM provides clear decision boundaries and feature importance",
                "performance": "Excellent performance on text classification tasks",
                "efficiency": "Fast training and inference, suitable for production",
                "robustness": "Handles various document types and lengths well",
                "scalability": "Can be easily retrained with new categories"
            },
            "alternatives_considered": {
                "BERT-based": "Higher accuracy but much slower and resource intensive",
                "Naive Bayes": "Faster but less accurate for complex documents",
                "Random Forest": "Good performance but less interpretable"
            },
            "feature_engineering": {
                "vectorization": "TF-IDF with 1-2 grams for context",
                "preprocessing": "Lowercase, stop word removal",
                "keyword_boost": "Domain-specific keyword matching for confidence boost"
            },
            "model_info": self.model_info if hasattr(self, 'model_info') else {}
        }
