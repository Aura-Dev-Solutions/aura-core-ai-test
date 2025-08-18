"""
Named Entity Recognition (NER) model for document analysis.
Extracts entities like persons, organizations, dates, amounts, etc.
"""

import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import spacy
from spacy.matcher import Matcher

from src.core.exceptions import NERExtractionError, ModelLoadError
from src.core.config import settings
from src.core.logging import log_performance
from src.core.models import NamedEntity
from .base import BaseNERModel


class DocumentNERModel(BaseNERModel):
    """
    Named Entity Recognition model for document analysis.
    
    Extracts entities:
    - PERSON: Names of people
    - ORG: Organizations, companies
    - DATE: Dates and time expressions
    - MONEY: Monetary amounts
    - EMAIL: Email addresses
    - PHONE: Phone numbers
    - CONTRACT_ID: Contract identifiers
    - LEGAL_REF: Legal references
    - TECH_TERM: Technical terminology
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the NER model.
        
        Args:
            model_name: spaCy model name to use
        """
        model_name = model_name or settings.ner_model
        super().__init__(model_name)
        
        self.nlp = None
        self.matcher = None
        
        # Custom patterns for domain-specific entities
        self.custom_patterns = {
            "EMAIL": [
                {"LIKE_EMAIL": True}
            ],
            "PHONE": [
                {"TEXT": {"REGEX": r"^\+?[\d\s\-\(\)]{10,}$"}}
            ],
            "CONTRACT_ID": [
                {"TEXT": {"REGEX": r"^(CONT|CONTRACT|CTR)[-_]?\d+$"}},
                {"TEXT": {"REGEX": r"^[A-Z]{2,4}-\d{4,6}$"}}
            ],
            "LEGAL_REF": [
                {"TEXT": {"REGEX": r"^(Art|Article|Sec|Section)\.?\s*\d+"}},
                {"TEXT": {"REGEX": r"^Ley\s+\d+"}},
                {"TEXT": {"REGEX": r"^GDPR|^LOPD"}}
            ],
            "TECH_TERM": [
                {"LOWER": {"IN": ["api", "rest", "json", "xml", "sql", "nosql", "ai", "ml", "nlp"]}},
                {"TEXT": {"REGEX": r"^[A-Z]{2,}$"}, "LENGTH": {">=": 3, "<=": 6}}
            ]
        }
        
        # Entity confidence thresholds
        self.confidence_thresholds = {
            "PERSON": 0.7,
            "ORG": 0.6,
            "DATE": 0.8,
            "MONEY": 0.9,
            "EMAIL": 0.95,
            "PHONE": 0.85,
            "CONTRACT_ID": 0.9,
            "LEGAL_REF": 0.8,
            "TECH_TERM": 0.6
        }
    
    async def load_model(self) -> None:
        """Load the spaCy NER model."""
        try:
            def _load_model():
                try:
                    # Try to load the specified model
                    nlp = spacy.load(self.model_name)
                except OSError:
                    # Fallback to English model
                    try:
                        nlp = spacy.load("en_core_web_sm")
                        self.logger.warning(
                            f"Model {self.model_name} not found, using en_core_web_sm"
                        )
                    except OSError:
                        raise ModelLoadError(
                            "No spaCy model found. Install with: python -m spacy download en_core_web_sm"
                        )
                
                # Add custom matcher
                matcher = Matcher(nlp.vocab)
                
                # Add custom patterns
                for label, patterns in self.custom_patterns.items():
                    matcher.add(label, patterns)
                
                return nlp, matcher
            
            self.nlp, self.matcher = await asyncio.get_event_loop().run_in_executor(None, _load_model)
            
            self.is_loaded = True
            self.model_info = {
                'model_name': self.model_name,
                'language': self.nlp.lang,
                'pipeline': list(self.nlp.pipe_names),
                'custom_patterns': len(self.custom_patterns)
            }
            
            self.logger.info(
                "NER model loaded successfully",
                model_name=self.model_name,
                language=self.nlp.lang,
                pipeline=self.nlp.pipe_names
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to load NER model",
                model_name=self.model_name,
                error=str(e)
            )
            raise ModelLoadError(f"Failed to load NER model: {str(e)}") from e
    
    @log_performance("ner_extraction")
    async def extract_entities(self, text: str) -> List[NamedEntity]:
        """
        Extract named entities from text.
        
        Args:
            text: Text to process
            
        Returns:
            List of extracted entities
        """
        await self.ensure_loaded()
        
        try:
            def _extract():
                # Process text with spaCy
                doc = self.nlp(text)
                
                entities = []
                
                # Extract standard spaCy entities
                for ent in doc.ents:
                    confidence = self._calculate_confidence(ent)
                    threshold = self.confidence_thresholds.get(ent.label_, 0.5)
                    
                    if confidence >= threshold:
                        entity = NamedEntity(
                            text=ent.text,
                            label=ent.label_,
                            start_char=ent.start_char,
                            end_char=ent.end_char,
                            confidence=confidence,
                            metadata={
                                'spacy_label': ent.label_,
                                'lemma': ent.lemma_ if hasattr(ent, 'lemma_') else None,
                                'pos': ent.root.pos_ if hasattr(ent.root, 'pos_') else None
                            }
                        )
                        entities.append(entity)
                
                # Extract custom pattern matches
                matches = self.matcher(doc)
                for match_id, start, end in matches:
                    label = self.nlp.vocab.strings[match_id]
                    span = doc[start:end]
                    
                    # Avoid duplicates with spaCy entities
                    if not any(e.start_char <= span.start_char < e.end_char or 
                             e.start_char < span.end_char <= e.end_char for e in entities):
                        
                        confidence = self.confidence_thresholds.get(label, 0.8)
                        
                        entity = NamedEntity(
                            text=span.text,
                            label=label,
                            start_char=span.start_char,
                            end_char=span.end_char,
                            confidence=confidence,
                            metadata={
                                'pattern_match': True,
                                'custom_label': label
                            }
                        )
                        entities.append(entity)
                
                # Extract additional domain-specific entities
                domain_entities = self._extract_domain_entities(text)
                entities.extend(domain_entities)
                
                # Sort by start position
                entities.sort(key=lambda x: x.start_char)
                
                return entities
            
            entities = await asyncio.get_event_loop().run_in_executor(None, _extract)
            
            return entities
            
        except Exception as e:
            self.logger.error(
                "NER extraction failed",
                text_length=len(text),
                error=str(e)
            )
            raise NERExtractionError(f"Failed to extract entities: {str(e)}") from e
    
    def _calculate_confidence(self, ent) -> float:
        """Calculate confidence score for an entity."""
        # Base confidence from entity type
        base_confidence = {
            'PERSON': 0.8,
            'ORG': 0.7,
            'DATE': 0.9,
            'TIME': 0.9,
            'MONEY': 0.95,
            'PERCENT': 0.9,
            'GPE': 0.7,  # Geopolitical entity
            'LOC': 0.7,  # Location
        }.get(ent.label_, 0.6)
        
        # Adjust based on entity length and context
        length_factor = min(1.0, len(ent.text) / 20)  # Longer entities are more reliable
        
        # Check if entity is all caps (might be acronym)
        caps_factor = 0.9 if ent.text.isupper() and len(ent.text) > 2 else 1.0
        
        # Final confidence
        confidence = base_confidence * (0.7 + 0.3 * length_factor) * caps_factor
        
        return min(1.0, confidence)
    
    def _extract_domain_entities(self, text: str) -> List[NamedEntity]:
        """Extract domain-specific entities using regex patterns."""
        entities = []
        
        # Email addresses (improved pattern)
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entity = NamedEntity(
                text=match.group(),
                label="EMAIL",
                start_char=match.start(),
                end_char=match.end(),
                confidence=0.95,
                metadata={'regex_pattern': 'email'}
            )
            entities.append(entity)
        
        # Phone numbers (international format)
        phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
        for match in re.finditer(phone_pattern, text):
            if len(match.group().replace(' ', '').replace('-', '').replace('.', '')) >= 10:
                entity = NamedEntity(
                    text=match.group(),
                    label="PHONE",
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.85,
                    metadata={'regex_pattern': 'phone'}
                )
                entities.append(entity)
        
        # Monetary amounts (multiple currencies)
        money_pattern = r'(\$|€|£|USD|EUR|GBP)\s*[\d,]+\.?\d*|\d+\s*(dollars?|euros?|pounds?)'
        for match in re.finditer(money_pattern, text, re.IGNORECASE):
            entity = NamedEntity(
                text=match.group(),
                label="MONEY",
                start_char=match.start(),
                end_char=match.end(),
                confidence=0.9,
                metadata={'regex_pattern': 'money'}
            )
            entities.append(entity)
        
        # Contract/Document IDs
        contract_pattern = r'\b(CONT|CONTRACT|CTR|DOC)[-_]?\d{3,}\b'
        for match in re.finditer(contract_pattern, text, re.IGNORECASE):
            entity = NamedEntity(
                text=match.group(),
                label="CONTRACT_ID",
                start_char=match.start(),
                end_char=match.end(),
                confidence=0.9,
                metadata={'regex_pattern': 'contract_id'}
            )
            entities.append(entity)
        
        # Dates (various formats)
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or DD/MM/YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD
            r'\b\d{1,2}\s+(de\s+)?(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+(de\s+)?\d{4}\b',  # Spanish dates
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'  # English dates
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = NamedEntity(
                    text=match.group(),
                    label="DATE",
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.8,
                    metadata={'regex_pattern': 'date'}
                )
                entities.append(entity)
        
        return entities
    
    async def extract_entities_batch(self, texts: List[str]) -> List[List[NamedEntity]]:
        """
        Extract entities from multiple texts.
        
        Args:
            texts: List of texts to process
            
        Returns:
            List of entity lists for each text
        """
        results = []
        for text in texts:
            entities = await self.extract_entities(text)
            results.append(entities)
        
        return results
    
    def get_entity_types(self) -> List[str]:
        """Get list of entity types this model can recognize."""
        spacy_labels = ["PERSON", "ORG", "DATE", "TIME", "MONEY", "PERCENT", "GPE", "LOC"]
        custom_labels = list(self.custom_patterns.keys())
        domain_labels = ["EMAIL", "PHONE", "CONTRACT_ID"]
        
        return sorted(set(spacy_labels + custom_labels + domain_labels))
    
    async def predict(self, input_data: str) -> List[Dict[str, Any]]:
        """
        Make predictions with the model (implements abstract method).
        
        Args:
            input_data: Text to process
            
        Returns:
            List of entities as dictionaries
        """
        entities = await self.extract_entities(input_data)
        return [
            {
                'text': entity.text,
                'label': entity.label,
                'start_char': entity.start_char,
                'end_char': entity.end_char,
                'confidence': entity.confidence,
                'metadata': entity.metadata
            }
            for entity in entities
        ]
    
    async def unload_model(self) -> None:
        """Unload the model from memory."""
        self.nlp = None
        self.matcher = None
        self.is_loaded = False
        
        self.logger.info("NER model unloaded")
    
    def get_model_justification(self) -> Dict[str, Any]:
        """Get justification for model selection."""
        return {
            "selected_approach": "spaCy + Custom Patterns + Regex",
            "justification": {
                "accuracy": "spaCy provides high-accuracy NER for standard entities",
                "customization": "Custom patterns for domain-specific entities",
                "performance": "Fast processing suitable for production use",
                "multilingual": "Supports multiple languages including Spanish",
                "extensibility": "Easy to add new entity types and patterns"
            },
            "alternatives_considered": {
                "BERT-NER": "Higher accuracy but much slower and resource intensive",
                "Stanford NER": "Good accuracy but less flexible for customization",
                "Regex-only": "Fast but limited accuracy for complex entities"
            },
            "entity_types": {
                "standard": "PERSON, ORG, DATE, MONEY, etc. from spaCy",
                "custom": "EMAIL, PHONE, CONTRACT_ID, LEGAL_REF, TECH_TERM",
                "domain_specific": "Patterns tailored for document analysis"
            },
            "confidence_strategy": {
                "base_confidence": "Entity-type specific base scores",
                "length_adjustment": "Longer entities get higher confidence",
                "context_awareness": "Considers surrounding text context"
            },
            "model_info": self.model_info if hasattr(self, 'model_info') else {}
        }
    
    async def analyze_entity_distribution(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze entity distribution across multiple texts."""
        all_entities = []
        
        for text in texts:
            entities = await self.extract_entities(text)
            all_entities.extend(entities)
        
        # Count by label
        label_counts = {}
        confidence_by_label = {}
        
        for entity in all_entities:
            label = entity.label
            label_counts[label] = label_counts.get(label, 0) + 1
            
            if label not in confidence_by_label:
                confidence_by_label[label] = []
            confidence_by_label[label].append(entity.confidence)
        
        # Calculate statistics
        stats = {}
        for label, count in label_counts.items():
            confidences = confidence_by_label[label]
            stats[label] = {
                'count': count,
                'avg_confidence': sum(confidences) / len(confidences),
                'min_confidence': min(confidences),
                'max_confidence': max(confidences)
            }
        
        return {
            'total_entities': len(all_entities),
            'unique_labels': len(label_counts),
            'label_distribution': stats,
            'most_common': max(label_counts.items(), key=lambda x: x[1]) if label_counts else None
        }
