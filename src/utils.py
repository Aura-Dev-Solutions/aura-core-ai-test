import re
import os
import json
from typing import Set

def sanitize_text(text: str) -> str:
    if not text:
        return ""
    spanish_pain_pattern = r"\(esto es un pain point importante\)"
    text = re.sub(spanish_pain_pattern, "[Urgent Pain Point]", text, flags=re.IGNORECASE)
    text = text.replace("## Insight", "").replace("## Participant Quote", "")
    text = " ".join(text.split())
    text = re.sub(r"^[-\*\s]+", "", text)
    return text.strip()

def get_incremental_files(directory: str, processed_log: str = "processed_files.json") -> Set[str]:
    if not os.path.exists(processed_log):
        return set(os.listdir(directory))
    
    with open(processed_log, 'r') as f:
        processed = set(json.load(f))

    current_files = set(os.listdir(directory))
    return current_files - processed

def update_processed_log(new_files: Set[str], processed_log: str = "processed_files.json"):
    processed = set()
    if os.path.exists(processed_log):
        with open(processed_log, 'r') as f:
            processed = set(json.load(f))
            
    processed.update(new_files)
    with open(processed_log, 'w') as f:
        json.dump(list(processed), f)

def extract_product_area_from_text(text: str, valid_areas: Set[str]) -> str:
    text = text.lower()
    mapping = {
        "billing": ["payment", "invoice", "credit card", "billing"],
        "search": ["filter", "find", "query", "results"],
        "mobile": ["ios", "android", "phone", "app store"],
        "reporting": ["chart", "dashboard", "analytics", "export"],
        "integrations": ["api", "sso", "connected apps", "okta"]
    }
    for area, keywords in mapping.items():
        if any(kw in text for kw in keywords):
            return area
            
    return "general"