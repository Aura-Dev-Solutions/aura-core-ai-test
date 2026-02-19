import pandas as pd
import json
import os
from typing import List, Dict
from .models import ProductFeature, Interview, SessionNote

class ResearchIngestor:
    
    def __init__(self, state_file: str = "pipeline_state.json"):
        self.state_file = state_file
        self.processed_files = self._load_state()

    def _load_state(self) -> List[str]:
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f).get("processed_files", [])
        return []

    def _save_state(self, filename: str):
        self.processed_files.append(filename)
        with open(self.state_file, 'w') as f:
            json.dump({"processed_files": self.processed_files}, f)

    def ingest_backlog(self, file_path: str) -> List[ProductFeature]:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return [ProductFeature(**item) for item in data]

    def ingest_interviews(self, file_path: str) -> List[Interview]:
        if file_path in self.processed_files:
            return []

        with open(file_path, 'r') as f:
            data = json.load(f)
        
        interviews = []
        for item in data:
            if item.get("duration_minutes") == "":
                item["duration_minutes"] = None
                
            interviews.append(Interview(**item))
        
        self._save_state(file_path)
        return interviews

    def ingest_notes(self, file_path: str) -> List[SessionNote]:
        if file_path in self.processed_files:
            return []
        df = pd.read_csv(file_path, keep_default_na=False)
        df = df.replace(r'^\s*$', None, regex=True)
        
        if 'note_type' in df.columns:
            df['note_type'] = df['note_type'].str.replace('insigth', 'insight', case=False)
        
        notes = []
        for _, row in df.iterrows():
            data = row.to_dict()
            if data.get('product_area_tag') is None:
                data['product_area_tag'] = "unclassified"
                
            notes.append(SessionNote(**data))
            
        self._save_state(file_path)
        return notes