from typing import List, Tuple
from collections import defaultdict

KEYWORDS = {
    "invoice": [
        "invoice", "amount", "due date", "payment", "total", "bill", "receipt", "balance due",
        "factura", "importe", "fecha de vencimiento", "pago", "total", "recibo", "saldo pendiente"
    ],
    "contract": [
        "agreement", "party", "term", "clause", "signature", "contract", "obligation", "conditions",
        "contrato", "acuerdo", "parte", "plazo", "cláusula", "firma", "obligación", "condiciones"
    ],
    "resume": [
        "experience", "education", "skills", "projects", "summary", "curriculum vitae", "work history",
        "experiencia", "educación", "habilidades", "proyectos", "resumen", "currículum", "historial laboral"
    ],
    "report": [
        "report", "findings", "analysis", "results", "conclusion", "executive summary",
        "reporte", "informe", "hallazgos", "análisis", "resultados", "conclusión", "resumen ejecutivo"
    ],
    "manual": [
        "manual", "instructions", "guide", "steps", "procedure", "reference",
        "manual", "instrucciones", "guía", "pasos", "procedimiento", "referencia"
    ],
    "government": [
        "government", "ministry", "department", "official", "license", "permit", "certificate",
        "public administration", "tax office", "regulation", "policy", "agency", "immigration",
        "gobierno", "ministerio", "secretaría", "dependencia", "oficial", "licencia", "permiso",
        "certificado", "administración pública", "hacienda", "reglamento", "política", "instituto",
        "delegación", "inmigración"
    ]
}

def classify_text(text: str) -> Tuple[List[str], List[float]]:
    scores = defaultdict(float)
    lower = text.lower()
    for label, words in KEYWORDS.items():
        for w in words:
            if w in lower:
                scores[label] += 1.0
    if not scores:
        return (["unknown"], [1.0])
    labels = list(scores.keys())
    vals = list(scores.values())
    total = sum(vals)
    probs = [v/total for v in vals]
    return labels, probs
