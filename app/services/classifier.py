def classify(text: str) -> str:
    if "invoice" in text.lower():
        return "invoice"
    elif "contract" in text.lower():
        return "contract"
    else:
        return "other"
