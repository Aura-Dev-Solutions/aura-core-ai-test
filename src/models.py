from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import dateparser
from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationInfo


class ProductFeature(BaseModel):
    model_config = ConfigDict(extra="ignore")

    feature_id: str
    title: str
    description: str
    product_area: str
    status: str
    priority: str
    tags: List[str] = Field(default_factory=list)
    release_date: Optional[str] = None
    quarter: Optional[str] = None


class ProductAreaAssignment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    product_area: str
    confidence: float = Field(ge=0.0, le=1.0)
    method: str = "keyword_scoring"
    evidence: List[str] = Field(default_factory=list)


class Interview(BaseModel):
    model_config = ConfigDict(extra="ignore")

    interview_id: str
    date: datetime
    participant_id: str
    participant_role: Optional[str] = None
    interviewer: str
    interview_type: str
    product_areas_discussed: List[str] = Field(default_factory=list)
    transcript: str
    duration_minutes: Optional[int] = 0
    recording_quality: Optional[str] = None
    taxonomy: List[ProductAreaAssignment] = Field(default_factory=list)
    participant_sentiment: Optional[str] = None  # positive|negative|neutral|mixed
    participant_sentiment_score: Optional[float] = None

    @field_validator("date", mode="before")
    @classmethod
    def parse_natural_date(cls, v):
        if isinstance(v, datetime):
            return v
        if not v:
            return datetime.now()
        parsed = dateparser.parse(str(v))
        return parsed or datetime.now()

    @field_validator("duration_minutes", mode="before")
    @classmethod
    def handle_empty_duration(cls, v):
        if v == "" or v is None:
            return 0
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

    @field_validator("transcript", mode="before")
    @classmethod
    def handle_missing_transcript(cls, v):
        return (v or "").strip()


class SessionNote(BaseModel):
    model_config = ConfigDict(extra="ignore")

    note_id: str
    created_at: datetime
    researcher: Optional[str] = "Unknown"
    related_interview_id: Optional[str] = None
    participant_id: Optional[str] = None
    note_type: str
    content: str
    product_area_tag: Optional[str] = "unclassified"
    priority: Optional[str] = "low"
    sentiment: Optional[str] = "neutral"
    source_batch: Optional[int] = None

    taxonomy: List[ProductAreaAssignment] = Field(default_factory=list)

    @field_validator("created_at", mode="before")
    @classmethod
    def parse_note_date(cls, v):
        if isinstance(v, datetime):
            return v
        if not v:
            return datetime.now()
        parsed = dateparser.parse(str(v))
        return parsed or datetime.now()

    @field_validator("note_type", mode="before")
    @classmethod
    def fix_typos(cls, v):
        if not v:
            return "observation"
        mapping = {
            "insigth": "insight",
            "pain point": "pain_point",
            "feature request": "feature_request",
            "request": "feature_request",
        }
        key = str(v).lower().strip()
        return mapping.get(key, key)

    @field_validator("sentiment", mode="before")
    @classmethod
    def normalize_sentiment(cls, v):
        if not v:
            return "neutral"
        key = str(v).lower().strip()
        mapping = {
            "positive": "positive",
            "pos": "positive",
            "negative": "negative",
            "neg": "negative",
            "neutral": "neutral",
            "mixed": "mixed",
        }
        return mapping.get(key, "neutral")

    @field_validator("product_area_tag", "related_interview_id", "participant_id", mode="before")
    @classmethod
    def handle_none_and_nan(cls, v, info: ValidationInfo):
        # Fixes None/NaN/empty strings robustly per-field (Pydantic v2).
        if v is None or (isinstance(v, float) and str(v) == "nan") or v == "":
            if info.field_name in {"related_interview_id", "participant_id"}:
                return None
            if info.field_name == "product_area_tag":
                return "unclassified"
        return str(v).strip() if isinstance(v, str) else str(v)


class BacklogItemRef(BaseModel):
    model_config = ConfigDict(extra="ignore")

    feature_id: str
    title: str
    status: str
    score: Optional[float] = None
    match_basis: Optional[str] = None 


class ProductAreaIntelligence(BaseModel):
    model_config = ConfigDict(extra="ignore")

    product_area: str
    mentions_total: int
    mentions_by_source: Dict[str, int]
    researcher_sentiment_distribution: Dict[str, int]
    participant_sentiment_distribution: Dict[str, int]
    key_themes: List[str] = Field(default_factory=list)
    pain_points: List[str] = Field(default_factory=list)
    feature_requests: List[str] = Field(default_factory=list)
    related_backlog_items: Dict[str, List[BacklogItemRef]] = Field(default_factory=dict)
    matched_backlog_from_requests: List[BacklogItemRef] = Field(default_factory=list)

    gaps: List[str] = Field(default_factory=list)
