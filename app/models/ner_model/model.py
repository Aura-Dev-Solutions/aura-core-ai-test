import spacy
from spacy.tokens import DocBin

# Datos de entrenamiento en inglés
TRAIN_DATA = [
    ("QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934 For the quarterly period ended March 31, 2022", {
        "entities": [
            (0, 16, "REPORT_TYPE"),
            (46, 82, "LAW"),
            (118, 130, "DATE")
        ]
    }),
    ("CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS (Unaudited)", {
        "entities": [(0, 48, "REPORT_TYPE")]
    }),
    ("CONDENSED CONSOLIDATED BALANCE SHEETS (Unaudited)", {
        "entities": [(0, 38, "REPORT_TYPE")]
    }),
    ("CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS (Unaudited)", {
        "entities": [(0, 48, "REPORT_TYPE")]
    }),
    ("Prepared by John Smith for Acme Corp on April 15, 2023", {
        "entities": [
            (13, 23, "PERSON"),
            (28, 37, "ORG"),
            (41, 55, "DATE")
        ]
    }),
    ("Submitted under the Securities Act of 1933 in New York", {
        "entities": [
            (20, 46, "LAW"),
            (50, 58, "GPE")
        ]
    }),
    ("December 30, 2023", {
        "entities": [
            (0, 16, "DATE"),
        ]
    }),
    
]

nlp = spacy.blank("en")  # Modelo vacío para inglés
db = DocBin()

for text, annotations in TRAIN_DATA:
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in annotations["entities"]:
        span = doc.char_span(start, end, label=label)
        if span:
            ents.append(span)
    doc.ents = ents
    db.add(doc)

# Guardar los datos en disco
db.to_disk("train_ner.spacy")
print("Datos de entrenamiento guardados como train_ner.spacy")