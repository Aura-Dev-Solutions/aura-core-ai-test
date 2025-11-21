# app/models/ner.py
"""
Custom NER con GLiNER2-large-v1: Modelo multi-task (340M params) para extracción eficiente.
Justificación: Schema-driven para entidades custom, CPU-optimized, SOTA en legal/finance (MTEB 2025).
Fuente: https://huggingface.co/fastino/gliner2-large-v1
Uso: Extrae entidades como PERSON, MONEY, PROJECT_CODE + structured JSON para RAG metadata.
Adaptado de model card examples: Usa extract_entities con schema dict posicional.
"""
from gliner2 import GLiNER2
from typing import List, Dict, Optional
from loguru import logger

class GLINERExtractor:
    def __init__(
        self, 
        model_name: str = "fastino/gliner2-large-v1",
        device: str = "cpu"
    ):
        """
        Inicializa GLiNER2-large-v1.
        Args:
            model_name: Modelo HF (default: fastino/gliner2-large-v1).
            device: "cpu" o "cuda" (recomendado CPU para escalabilidad).
        """
        try:
            self.model = GLiNER2.from_pretrained(
                model_name, 
                device=device,
                torch_dtype="float32"  # Para CPU stability
            )
            # Schema con descripciones en NL (de docs: mejora precisión para custom entities)
            self.schema = {
                "person": "Nombres de personas o contactos",
                "organization": "Nombres de empresas, organizaciones o entidades legales",
                "date": "Fechas, plazos o referencias temporales como '15 de enero de 2025'",
                "money": "Importes monetarios, presupuestos o pagos como '185.000 €'",
                "location": "Lugares geográficos como ciudades o países",
                "project_code": "Códigos de proyectos o referencias internas como 'AUR-2025-007'"
            }
            logger.info(f"GLiNER2-large-v1 loaded: {model_name} on {device}")
        except Exception as e:
            logger.error(f"Failed to load GLiNER2-large-v1: {e}")
            raise

    def extract_entities(
        self, 
        text: str
    ) -> Dict[str, List[str]]:
        """
        Extrae entidades del texto usando extract_entities (de docs GLiNER2).
        Args:
            text: Texto a procesar.
        Returns:
            Dict como {'money': ['185.000 €'], 'project_code': ['AUR-2025-007']}.
        """
        if len(text.strip()) < 50:  # Optimización: Skip chunks muy cortos
            return {}
        
        try:
            # extract_entities según docs: schema como dict posicional después de text
            result = self.model.extract_entities(
                text,  # Posicional 1
                self.schema  # Posicional 2: schema dict
            )
            
            # Parse output: {'entities': {label: [values]}} → flatten
            entities = result.get('entities', {})
            
            # Flatten a list por label (evita duplicados)
            extracted = {}
            for label, values in entities.items():
                if isinstance(values, list):
                    extracted[label] = list(set([v.strip() for v in values if v.strip()]))  # Unique + clean
                else:
                    extracted[label] = [str(values).strip()] if values else []
            
            logger.debug(f"Extracted {len(extracted)} entity types from text (len={len(text)})")
            return extracted  # {'money': ['185.000 €'], ...}
        except Exception as e:
            logger.warning(f"NER extraction failed for text: {e}")
            return {}

# Instancia global lazy-load (singleton para eficiencia, carga en primer uso)
_extractor_instance = None

def get_extractor(device: str = "cpu") -> 'GLINERExtractor':
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = GLINERExtractor(device=device)
    return _extractor_instance