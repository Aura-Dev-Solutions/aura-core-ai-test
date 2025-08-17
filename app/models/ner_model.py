import spacy

# Carga tu modelo entrenado (ajusta la ruta si es necesario)
nlp = spacy.load("app/models/ner_model/model-best")

def extract_entities(text: str):
    """
    Extrae entidades nombradas usando un modelo spaCy personalizado.
    :param text: Texto a analizar
    :return: lista de tuplas (entidad, tipo)
    """
    doc = nlp(text)
    return [(str(ent.text), str(ent.label_)) for ent in doc.ents]