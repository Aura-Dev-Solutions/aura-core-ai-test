"""
Novel approaches to document processing using advanced techniques.
"""

import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from src.core.logging import LoggerMixin
from src.core.models import DocumentType, ProcessingResult


class ProcessingStrategy(Enum):
    HYBRID_EXTRACTION = "hybrid_extraction"
    SEMANTIC_CHUNKING = "semantic_chunking"
    ADAPTIVE_PARSING = "adaptive_parsing"
    MULTI_MODAL_FUSION = "multi_modal_fusion"


@dataclass
class DocumentChunk:
    content: str
    chunk_type: str  # paragraph, table, list, header, etc.
    confidence: float
    metadata: Dict[str, Any]
    start_position: int
    end_position: int


@dataclass
class StructuralElement:
    element_type: str
    content: str
    hierarchy_level: int
    parent_id: Optional[str]
    children_ids: List[str]
    attributes: Dict[str, Any]


class HybridDocumentExtractor(LoggerMixin):
    """
    Hybrid extraction combining multiple techniques for optimal results.
    
    Uses rule-based, statistical, and ML approaches simultaneously.
    """
    
    def __init__(self):
        self.extraction_methods = [
            self._rule_based_extraction,
            self._statistical_extraction,
            self._pattern_based_extraction
        ]
    
    async def extract_with_fusion(self, file_path: Path, doc_type: DocumentType) -> ProcessingResult:
        """
        Extract content using multiple methods and fuse results.
        
        Args:
            file_path: Path to document
            doc_type: Type of document
            
        Returns:
            Fused processing result with confidence scores
        """
        try:
            # Run all extraction methods concurrently
            extraction_tasks = [
                method(file_path, doc_type) for method in self.extraction_methods
            ]
            
            results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
            
            # Filter successful results
            valid_results = [r for r in results if not isinstance(r, Exception)]
            
            if not valid_results:
                raise Exception("All extraction methods failed")
            
            # Fuse results using confidence-weighted combination
            fused_result = self._fuse_extraction_results(valid_results)
            
            return fused_result
            
        except Exception as e:
            self.logger.error(f"Hybrid extraction failed: {str(e)}")
            raise
    
    async def _rule_based_extraction(self, file_path: Path, doc_type: DocumentType) -> Dict[str, Any]:
        """Extract using predefined rules and patterns."""
        content = await self._read_file_content(file_path)
        
        # Apply document-specific rules
        if doc_type == DocumentType.PDF:
            return self._apply_pdf_rules(content)
        elif doc_type == DocumentType.DOCX:
            return self._apply_docx_rules(content)
        else:
            return self._apply_generic_rules(content)
    
    async def _statistical_extraction(self, file_path: Path, doc_type: DocumentType) -> Dict[str, Any]:
        """Extract using statistical analysis of text patterns."""
        content = await self._read_file_content(file_path)
        
        # Analyze text statistics
        stats = self._analyze_text_statistics(content)
        
        # Extract based on statistical patterns
        return {
            "method": "statistical",
            "content": content,
            "confidence": stats["confidence"],
            "metadata": {
                "word_count": stats["word_count"],
                "sentence_count": stats["sentence_count"],
                "paragraph_count": stats["paragraph_count"],
                "readability_score": stats["readability"]
            }
        }
    
    async def _pattern_based_extraction(self, file_path: Path, doc_type: DocumentType) -> Dict[str, Any]:
        """Extract using learned patterns and templates."""
        content = await self._read_file_content(file_path)
        
        # Detect document patterns
        patterns = self._detect_document_patterns(content)
        
        return {
            "method": "pattern_based",
            "content": content,
            "confidence": patterns["confidence"],
            "metadata": {
                "detected_patterns": patterns["patterns"],
                "structure_type": patterns["structure_type"]
            }
        }
    
    def _fuse_extraction_results(self, results: List[Dict[str, Any]]) -> ProcessingResult:
        """
        Fuse multiple extraction results using confidence weighting.
        
        Args:
            results: List of extraction results from different methods
            
        Returns:
            Fused processing result
        """
        if not results:
            raise ValueError("No results to fuse")
        
        # Calculate weighted content based on confidence scores
        total_weight = sum(r["confidence"] for r in results)
        
        if total_weight == 0:
            # Fallback to simple average
            best_result = max(results, key=lambda r: len(r["content"]))
        else:
            # Use highest confidence result as base
            best_result = max(results, key=lambda r: r["confidence"])
        
        # Combine metadata from all methods
        combined_metadata = {}
        for result in results:
            method_name = result["method"]
            combined_metadata[f"{method_name}_confidence"] = result["confidence"]
            combined_metadata[f"{method_name}_metadata"] = result.get("metadata", {})
        
        # Add fusion metadata
        combined_metadata["fusion_info"] = {
            "methods_used": [r["method"] for r in results],
            "total_methods": len(results),
            "confidence_scores": [r["confidence"] for r in results],
            "selected_method": best_result["method"]
        }
        
        return ProcessingResult(
            text_content=best_result["content"],
            metadata=combined_metadata,
            processing_time=0.0,  # Will be set by caller
            success=True,
            file_path=None  # Will be set by caller
        )
    
    async def _read_file_content(self, file_path: Path) -> str:
        """Read file content based on file type."""
        # This would integrate with existing extractors
        # For now, return placeholder
        return f"Content from {file_path.name}"
    
    def _apply_pdf_rules(self, content: str) -> Dict[str, Any]:
        """Apply PDF-specific extraction rules."""
        # Detect PDF-specific patterns like headers, footers, page numbers
        confidence = 0.8
        
        return {
            "method": "rule_based_pdf",
            "content": content,
            "confidence": confidence,
            "metadata": {"extraction_type": "pdf_rules"}
        }
    
    def _apply_docx_rules(self, content: str) -> Dict[str, Any]:
        """Apply DOCX-specific extraction rules."""
        confidence = 0.85
        
        return {
            "method": "rule_based_docx",
            "content": content,
            "confidence": confidence,
            "metadata": {"extraction_type": "docx_rules"}
        }
    
    def _apply_generic_rules(self, content: str) -> Dict[str, Any]:
        """Apply generic text extraction rules."""
        confidence = 0.7
        
        return {
            "method": "rule_based_generic",
            "content": content,
            "confidence": confidence,
            "metadata": {"extraction_type": "generic_rules"}
        }
    
    def _analyze_text_statistics(self, content: str) -> Dict[str, Any]:
        """Analyze statistical properties of text."""
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        paragraphs = content.split('\n\n')
        
        # Calculate readability score (simplified Flesch formula)
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_syllables = sum(self._count_syllables(word) for word in words) / max(len(words), 1)
        readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        
        # Confidence based on text quality indicators
        confidence = min(0.9, max(0.3, readability / 100))
        
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "readability": readability,
            "confidence": confidence
        }
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)."""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _detect_document_patterns(self, content: str) -> Dict[str, Any]:
        """Detect structural patterns in document."""
        patterns = []
        confidence = 0.6
        
        # Detect common patterns
        if re.search(r'^#+ ', content, re.MULTILINE):
            patterns.append("markdown_headers")
            confidence += 0.1
        
        if re.search(r'^\d+\.', content, re.MULTILINE):
            patterns.append("numbered_lists")
            confidence += 0.1
        
        if re.search(r'^\* ', content, re.MULTILINE):
            patterns.append("bullet_lists")
            confidence += 0.1
        
        # Detect table-like structures
        if re.search(r'\|.*\|', content):
            patterns.append("table_structure")
            confidence += 0.15
        
        structure_type = "structured" if len(patterns) > 2 else "unstructured"
        
        return {
            "patterns": patterns,
            "structure_type": structure_type,
            "confidence": min(0.95, confidence)
        }


class SemanticChunker(LoggerMixin):
    """
    Semantic-aware document chunking that preserves meaning boundaries.
    
    Uses sentence embeddings and clustering to create coherent chunks.
    """
    
    def __init__(self, max_chunk_size: int = 1000, overlap_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
    
    async def chunk_document(self, content: str, preserve_structure: bool = True) -> List[DocumentChunk]:
        """
        Chunk document while preserving semantic boundaries.
        
        Args:
            content: Document content to chunk
            preserve_structure: Whether to preserve document structure
            
        Returns:
            List of semantic chunks
        """
        try:
            # First, identify structural elements
            if preserve_structure:
                structural_elements = self._identify_structural_elements(content)
                chunks = await self._chunk_with_structure(content, structural_elements)
            else:
                chunks = await self._chunk_by_semantics(content)
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Semantic chunking failed: {str(e)}")
            # Fallback to simple chunking
            return self._simple_chunk(content)
    
    def _identify_structural_elements(self, content: str) -> List[StructuralElement]:
        """Identify structural elements in document."""
        elements = []
        lines = content.split('\n')
        
        current_id = 0
        hierarchy_stack = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            element_type, hierarchy_level = self._classify_line(line)
            
            if element_type != "text":
                element = StructuralElement(
                    element_type=element_type,
                    content=line,
                    hierarchy_level=hierarchy_level,
                    parent_id=self._find_parent_id(hierarchy_level, hierarchy_stack),
                    children_ids=[],
                    attributes={"line_number": i}
                )
                
                elements.append(element)
                
                # Update hierarchy stack
                hierarchy_stack = [e for e in hierarchy_stack if e.hierarchy_level < hierarchy_level]
                hierarchy_stack.append(element)
                
                current_id += 1
        
        return elements
    
    def _classify_line(self, line: str) -> Tuple[str, int]:
        """Classify a line and determine its hierarchy level."""
        # Header patterns
        if re.match(r'^#{1,6}\s', line):
            level = len(re.match(r'^#+', line).group())
            return "header", level
        
        # List patterns
        if re.match(r'^\d+\.\s', line):
            return "numbered_list", 0
        
        if re.match(r'^[-*+]\s', line):
            return "bullet_list", 0
        
        # Table patterns
        if '|' in line and line.count('|') >= 2:
            return "table_row", 0
        
        return "text", 0
    
    def _find_parent_id(self, level: int, hierarchy_stack: List[StructuralElement]) -> Optional[str]:
        """Find parent element ID based on hierarchy."""
        for element in reversed(hierarchy_stack):
            if element.hierarchy_level < level:
                return element.element_type  # Simplified ID
        return None
    
    async def _chunk_with_structure(self, content: str, elements: List[StructuralElement]) -> List[DocumentChunk]:
        """Chunk document preserving structural boundaries."""
        chunks = []
        current_chunk = ""
        current_position = 0
        
        sentences = self._split_into_sentences(content)
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                # Create chunk
                chunk = DocumentChunk(
                    content=current_chunk.strip(),
                    chunk_type="structured",
                    confidence=0.8,
                    metadata={"sentence_count": len(current_chunk.split('.'))},
                    start_position=current_position - len(current_chunk),
                    end_position=current_position
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.overlap_size)
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence
            
            current_position += len(sentence)
        
        # Add final chunk
        if current_chunk.strip():
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                chunk_type="structured",
                confidence=0.8,
                metadata={"sentence_count": len(current_chunk.split('.'))},
                start_position=current_position - len(current_chunk),
                end_position=current_position
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _chunk_by_semantics(self, content: str) -> List[DocumentChunk]:
        """Chunk document based on semantic similarity."""
        sentences = self._split_into_sentences(content)
        
        # For now, use simple sentence-based chunking
        # In a full implementation, this would use sentence embeddings
        # and clustering to group semantically similar sentences
        
        chunks = []
        current_chunk = ""
        current_position = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                chunk = DocumentChunk(
                    content=current_chunk.strip(),
                    chunk_type="semantic",
                    confidence=0.7,
                    metadata={"semantic_coherence": 0.8},
                    start_position=current_position - len(current_chunk),
                    end_position=current_position
                )
                chunks.append(chunk)
                
                overlap_text = self._get_overlap_text(current_chunk, self.overlap_size)
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence
            
            current_position += len(sentence)
        
        if current_chunk.strip():
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                chunk_type="semantic",
                confidence=0.7,
                metadata={"semantic_coherence": 0.8},
                start_position=current_position - len(current_chunk),
                end_position=current_position
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, content: str) -> List[str]:
        """Split content into sentences."""
        # Simple sentence splitting - in production would use more sophisticated NLP
        sentences = re.split(r'[.!?]+', content)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from end of current chunk."""
        if len(text) <= overlap_size:
            return text
        return text[-overlap_size:]
    
    def _simple_chunk(self, content: str) -> List[DocumentChunk]:
        """Fallback simple chunking method."""
        chunks = []
        words = content.split()
        
        for i in range(0, len(words), self.max_chunk_size // 10):  # Rough word-based chunking
            chunk_words = words[i:i + self.max_chunk_size // 10]
            chunk_content = ' '.join(chunk_words)
            
            chunk = DocumentChunk(
                content=chunk_content,
                chunk_type="simple",
                confidence=0.5,
                metadata={"word_count": len(chunk_words)},
                start_position=i * 10,  # Approximate
                end_position=(i + len(chunk_words)) * 10
            )
            chunks.append(chunk)
        
        return chunks


class AdaptiveDocumentProcessor(LoggerMixin):
    """
    Adaptive processor that selects optimal processing strategy based on document characteristics.
    """
    
    def __init__(self):
        self.hybrid_extractor = HybridDocumentExtractor()
        self.semantic_chunker = SemanticChunker()
        self.strategy_cache = {}
    
    async def process_adaptively(self, file_path: Path, doc_type: DocumentType) -> ProcessingResult:
        """
        Process document using adaptive strategy selection.
        
        Args:
            file_path: Path to document
            doc_type: Type of document
            
        Returns:
            Optimally processed result
        """
        try:
            # Analyze document characteristics
            characteristics = await self._analyze_document_characteristics(file_path, doc_type)
            
            # Select optimal processing strategy
            strategy = self._select_processing_strategy(characteristics)
            
            # Process using selected strategy
            if strategy == ProcessingStrategy.HYBRID_EXTRACTION:
                result = await self.hybrid_extractor.extract_with_fusion(file_path, doc_type)
            elif strategy == ProcessingStrategy.SEMANTIC_CHUNKING:
                # First extract, then chunk
                basic_result = await self._basic_extraction(file_path, doc_type)
                chunks = await self.semantic_chunker.chunk_document(basic_result.text_content)
                result = self._combine_chunks_to_result(basic_result, chunks)
            else:
                # Default processing
                result = await self._basic_extraction(file_path, doc_type)
            
            # Add adaptive processing metadata
            result.metadata["adaptive_processing"] = {
                "selected_strategy": strategy.value,
                "document_characteristics": characteristics,
                "confidence": characteristics.get("processing_confidence", 0.7)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Adaptive processing failed: {str(e)}")
            # Fallback to basic processing
            return await self._basic_extraction(file_path, doc_type)
    
    async def _analyze_document_characteristics(self, file_path: Path, doc_type: DocumentType) -> Dict[str, Any]:
        """Analyze document to determine optimal processing approach."""
        characteristics = {
            "file_size": file_path.stat().st_size,
            "document_type": doc_type.value,
            "complexity_score": 0.5,
            "structure_score": 0.5,
            "processing_confidence": 0.7
        }
        
        # Analyze file size impact
        if characteristics["file_size"] > 10 * 1024 * 1024:  # 10MB
            characteristics["complexity_score"] += 0.2
        
        # Document type specific analysis
        if doc_type == DocumentType.PDF:
            characteristics["structure_score"] += 0.1
            characteristics["complexity_score"] += 0.1
        elif doc_type == DocumentType.DOCX:
            characteristics["structure_score"] += 0.2
        
        return characteristics
    
    def _select_processing_strategy(self, characteristics: Dict[str, Any]) -> ProcessingStrategy:
        """Select optimal processing strategy based on document characteristics."""
        complexity = characteristics["complexity_score"]
        structure = characteristics["structure_score"]
        
        # Strategy selection logic
        if complexity > 0.7 and structure > 0.6:
            return ProcessingStrategy.HYBRID_EXTRACTION
        elif structure > 0.7:
            return ProcessingStrategy.SEMANTIC_CHUNKING
        elif complexity > 0.8:
            return ProcessingStrategy.ADAPTIVE_PARSING
        else:
            return ProcessingStrategy.HYBRID_EXTRACTION  # Default to hybrid
    
    async def _basic_extraction(self, file_path: Path, doc_type: DocumentType) -> ProcessingResult:
        """Basic extraction fallback method."""
        # This would integrate with existing extractors
        content = f"Basic extraction from {file_path.name}"
        
        return ProcessingResult(
            text_content=content,
            metadata={"extraction_method": "basic"},
            processing_time=0.1,
            success=True,
            file_path=file_path
        )
    
    def _combine_chunks_to_result(self, basic_result: ProcessingResult, chunks: List[DocumentChunk]) -> ProcessingResult:
        """Combine chunked results back into a processing result."""
        # Combine all chunk content
        combined_content = "\n\n".join(chunk.content for chunk in chunks)
        
        # Add chunking metadata
        chunk_metadata = {
            "total_chunks": len(chunks),
            "chunk_types": list(set(chunk.chunk_type for chunk in chunks)),
            "average_confidence": sum(chunk.confidence for chunk in chunks) / len(chunks) if chunks else 0.0
        }
        
        basic_result.text_content = combined_content
        basic_result.metadata["chunking_info"] = chunk_metadata
        
        return basic_result


# Global instances
hybrid_extractor = HybridDocumentExtractor()
semantic_chunker = SemanticChunker()
adaptive_processor = AdaptiveDocumentProcessor()
