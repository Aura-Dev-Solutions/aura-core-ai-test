from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Iterable, Tuple, Optional

from .models import (
    SessionNote,
    Interview,
    ProductFeature,
    ProductAreaAssignment,
    ProductAreaIntelligence,
    BacklogItemRef,
)

_WORD_RE = re.compile(r"[a-z0-9]+", re.I)

_STOPWORDS = {
    "the","a","an","and","or","but","if","then","so","to","of","in","on","for","with","at","by","from",
    "is","are","was","were","be","been","being","it","this","that","these","those","as","we","i","you",
    "they","he","she","them","his","her","our","their","my","your","me","us","can","could","would","should",
    "not","no","yes","do","does","did","done","have","has","had","will","just","very","really","also","into",
    "about","over","under","more","less","when","where","what","why","how"
}

_MANUAL_AREA_PHRASES: Dict[str, List[str]] = {
    "billing": ["billing", "payment", "invoice", "credit card", "pricing", "subscription", "charge"],
    "search": ["search", "find", "filter", "query", "results", "sort", "keyword"],
    "mobile": ["ios", "android", "mobile", "phone", "app", "tablet", "native"],
    "reporting": ["export", "report", "pdf", "csv", "scheduled report", "download report"],
    "dashboard": ["dashboard", "chart", "analytics", "metrics", "kpi", "overview", "visualization", "graph"],
    "integrations": ["integration", "integrations", "webhook", "zapier", "salesforce", "slack", "connected apps", "sso", "okta"],
    "api": ["api", "endpoint", "token", "rate limit", "documentation", "auth", "swagger"],
    "user-management": ["user management", "permissions", "roles", "rbac", "access", "admin", "provisioning", "team"],
    "notifications": ["notification", "notifications", "alerts", "email alert", "push", "reminder", "notify"],
    "data-import": ["import", "upload", "csv upload", "ingest", "mapping", "data load", "migration"],
}

_AREA_SYNONYMS: Dict[str, str] = {
    "billing/payments": "billing",
    "payments": "billing",
    "payment": "billing",
    "invoicing": "billing",
    "dashboards": "dashboard",
    "analytics": "dashboard",
    "report": "reporting",
    "reports": "reporting",
    "export": "reporting",
    "dataimport": "data-import",
    "data import": "data-import",
    "imports": "data-import",
    "user management": "user-management",
    "user mgmt": "user-management",
    "usermanagement": "user-management",
    "integration": "integrations",
    "mobile app": "mobile",
    "ios app": "mobile",
    "android app": "mobile",
}

_POS_WORDS = {
    "love","like","liked","great","good","awesome","excellent","helpful","easy","smooth","fast","intuitive","clear","perfect",
    "works","working","convenient","nice","amazing"
}
_NEG_WORDS = {
    "hate","dislike","bad","awful","terrible","frustrating","frustrated","confusing","confused","hard","difficult","slow",
    "broken","issue","problem","bugs","bug","error","pain","annoying","can't","cannot","unable","worse","poor"
}

_PAIN_PATTERNS = [
    re.compile(r"\b(can't|cannot|unable to|hard to|difficult to|confus|frustrat|problem|issue|broken|bug|error|slow)\b", re.I),
]
_REQ_PATTERNS = [
    re.compile(r"\b(wish|would like|need to|need a|request|feature request|please add|could you|can we|it would be great|should have)\b", re.I),
]


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _ngrams(tokens: List[str], n: int) -> Iterable[str]:
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i : i + n])


def _sentiment_from_text(text: str) -> Tuple[str, float]:
    toks = _tokenize(text)
    pos = sum(1 for t in toks if t in _POS_WORDS)
    neg = sum(1 for t in toks if t in _NEG_WORDS)
    score = (pos - neg) / (pos + neg + 1.0)
    if score >= 0.2:
        return "positive", float(score)
    if score <= -0.2:
        return "negative", float(score)
    if pos > 0 and neg > 0:
        return "mixed", float(score)
    return "neutral", float(score)


def _extract_sentences(text: str, max_sentences: int = 200) -> List[str]:
    raw = (text or "").replace("\n", " ").strip()
    parts = re.split(r"(?<=[.!?])\s+", raw)
    out = [p.strip() for p in parts if len(p.strip()) >= 20]
    return out[:max_sentences]


def _dedupe_preserve(items: List[str], max_items: int) -> List[str]:
    seen = set()
    out = []
    for x in items:
        k = re.sub(r"\s+", " ", x.strip().lower())
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x.strip())
        if len(out) >= max_items:
            break
    return out


class ResearchProcessor:
    def __init__(self, backlog: List[ProductFeature]):
        self.backlog = backlog

        backlog_areas = sorted({self.normalize_area(f.product_area) for f in backlog if f.product_area})
        self.valid_areas = [a for a in backlog_areas if a] + ["unclassified"]

        self.area_lexicon: Dict[str, set[str]] = self._build_area_lexicon()

        self.backlog_by_area_status: Dict[str, Dict[str, List[ProductFeature]]] = defaultdict(lambda: defaultdict(list))
        for f in backlog:
            area = self.normalize_area(f.product_area) or "unclassified"
            status = (f.status or "unknown").lower().strip()
            self.backlog_by_area_status[area][status].append(f)

        self._backlog_search_index = [
            (
                f.feature_id,
                (f.title or ""),
                (f.status or "unknown").lower().strip(),
                self.normalize_area(f.product_area) or "unclassified",
                set(_tokenize((f.title or "") + " " + " ".join(f.tags or []))),
            )
            for f in backlog
        ]

    @staticmethod
    def normalize_area(area: Optional[str]) -> Optional[str]:
        if not area:
            return None
        a = str(area).strip().lower()
        a = re.sub(r"\s+", " ", a)
        if "/" in a:
            a = a.split("/")[0].strip()
        return _AREA_SYNONYMS.get(a, a)

    def _build_area_lexicon(self) -> Dict[str, set[str]]:
        lex = {a: set() for a in self.valid_areas}
        for f in self.backlog:
            area = self.normalize_area(f.product_area) or "unclassified"
            lex[area].update(_tokenize(f.title))
            lex[area].update(_tokenize(" ".join(f.tags or [])))
        for area, phrases in _MANUAL_AREA_PHRASES.items():
            if area not in lex:
                continue
            for ph in phrases:
                lex[area].update(_tokenize(ph))
        for area in lex:
            lex[area] = {t for t in lex[area] if len(t) >= 3 and t not in _STOPWORDS}
        return lex

    def _score_text_against_area(self, text: str, area: str) -> Tuple[int, List[str]]:
        text_l = (text or "").lower()
        toks = set(_tokenize(text_l))
        evidence = []

        overlap = toks.intersection(self.area_lexicon.get(area, set()))
        score = len(overlap)
        evidence.extend(list(sorted(overlap))[:8])

        for phrase in _MANUAL_AREA_PHRASES.get(area, []):
            if phrase.lower() in text_l:
                score += 3
                if phrase not in evidence:
                    evidence.append(phrase)

        return score, evidence[:10]

    def _classify_text(self, text: str, strong_areas: Optional[List[str]] = None, top_k: int = 2) -> List[ProductAreaAssignment]:
        strong_areas = [self.normalize_area(a) for a in (strong_areas or [])]
        strong_areas = [a for a in strong_areas if a]

        scored: List[Tuple[str, int, List[str]]] = []
        for area in self.valid_areas:
            if area == "unclassified":
                continue
            s, ev = self._score_text_against_area(text, area)
            if area in strong_areas:
                s += 10
                ev = (["explicit_tag"] + ev)[:10]
            scored.append((area, s, ev))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: max(top_k, 1)]

        if not top or top[0][1] <= 0:
            return [ProductAreaAssignment(product_area="unclassified", confidence=1.0, evidence=["no_keyword_signal"])]

        best = top[0][1]
        kept = [(a, s, ev) for a, s, ev in scored if s > 0 and s >= max(2, int(best * 0.6))]
        kept = kept[:top_k]

        total = sum(s for _, s, _ in kept) or 1
        return [
            ProductAreaAssignment(
                product_area=a,
                confidence=round(s / total, 3),
                method="keyword_scoring",
                evidence=ev,
            )
            for a, s, ev in kept
        ]

    def unify_taxonomy(self, notes: List[SessionNote]) -> List[SessionNote]:
        for note in notes:
            raw_tag = self.normalize_area(note.product_area_tag)
            strong = [raw_tag] if raw_tag and raw_tag != "unclassified" else []
            assignments = self._classify_text(note.content, strong_areas=strong, top_k=2)

            note.taxonomy = assignments
            note.product_area_tag = assignments[0].product_area if assignments else (raw_tag or "unclassified")

        return notes

    def classify_interviews(self, interviews: List[Interview]) -> List[Interview]:
        for iv in interviews:
            explicit = [self.normalize_area(a) for a in (iv.product_areas_discussed or [])]
            explicit = [a for a in explicit if a and a != "unclassified"]

            iv.taxonomy = self._classify_text(iv.transcript, strong_areas=explicit, top_k=2)

            label, score = _sentiment_from_text(iv.transcript)
            iv.participant_sentiment = label
            iv.participant_sentiment_score = score

            iv.product_areas_discussed = explicit

        return interviews

    def _extract_pain_points_from_text(self, text: str, max_items: int = 6) -> List[str]:
        hits = []
        for s in _extract_sentences(text):
            if any(p.search(s) for p in _PAIN_PATTERNS):
                hits.append(s)
        return _dedupe_preserve(hits, max_items=max_items)

    def _extract_requests_from_text(self, text: str, max_items: int = 6) -> List[str]:
        hits = []
        for s in _extract_sentences(text):
            if any(p.search(s) for p in _REQ_PATTERNS):
                hits.append(s)
        return _dedupe_preserve(hits, max_items=max_items)

    def _themes_from_texts(self, texts: List[str], top_k: int = 8) -> List[str]:
        toks = []
        for t in texts:
            toks.extend([w for w in _tokenize(t) if w not in _STOPWORDS and len(w) >= 3])
        if not toks:
            return []

        unigram = Counter(toks)
        bigram = Counter(_ngrams(toks, 2))

        candidates = []
        for bg, c in bigram.items():
            if c >= 2:
                candidates.append((bg, c * 2))
        for ug, c in unigram.items():
            candidates.append((ug, c))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in candidates[:top_k]]

    def _match_requests_to_backlog(self, requests: List[str], top_n: int = 5) -> List[BacklogItemRef]:
        if not requests:
            return []

        req_tokens = set()
        for r in requests:
            req_tokens.update([t for t in _tokenize(r) if t not in _STOPWORDS and len(t) >= 3])

        if not req_tokens:
            return []

        scored = []
        for fid, title, status, _area, feat_tokens in self._backlog_search_index:
            inter = req_tokens.intersection(feat_tokens)
            if not inter:
                continue
            score = len(inter) / (len(req_tokens.union(feat_tokens)) + 1e-9)
            scored.append((fid, title, status, score))

        scored.sort(key=lambda x: x[3], reverse=True)
        return [
            BacklogItemRef(feature_id=fid, title=title, status=status, score=round(score, 3), match_basis="token_overlap")
            for fid, title, status, score in scored[:top_n]
        ]

    def generate_intelligence_report(
        self,
        notes: List[SessionNote],
        interviews: Optional[List[Interview]] = None,
    ) -> List[ProductAreaIntelligence]:
        interviews = interviews or []

        notes_by_area = defaultdict(list)
        for n in notes:
            area = (n.taxonomy[0].product_area if n.taxonomy else self.normalize_area(n.product_area_tag)) or "unclassified"
            notes_by_area[area].append(n)

        interviews_by_area = defaultdict(list)
        for iv in interviews:
            area = (iv.taxonomy[0].product_area if iv.taxonomy else (iv.product_areas_discussed[0] if iv.product_areas_discussed else None)) or "unclassified"
            interviews_by_area[area].append(iv)

        reports: List[ProductAreaIntelligence] = []

        for area in self.valid_areas:
            if area == "unclassified":
                continue

            area_notes = notes_by_area.get(area, [])
            area_interviews = interviews_by_area.get(area, [])

            mentions_total = len(area_notes) + len(area_interviews)
            if mentions_total == 0:
                continue

            mentions_by_source = {"notes": len(area_notes), "interviews": len(area_interviews)}

            researcher_sent = Counter((n.sentiment or "neutral") for n in area_notes)
            participant_sent = Counter((iv.participant_sentiment or "neutral") for iv in area_interviews)

            combined_texts = [n.content for n in area_notes] + [iv.transcript for iv in area_interviews]
            themes = self._themes_from_texts(combined_texts, top_k=8)

            pain_points = []
            feature_requests = []

            for n in area_notes:
                if n.note_type == "pain_point":
                    pain_points.append(n.content)
                if n.note_type == "feature_request":
                    feature_requests.append(n.content)

            for iv in area_interviews:
                pain_points.extend(self._extract_pain_points_from_text(iv.transcript, max_items=3))
                feature_requests.extend(self._extract_requests_from_text(iv.transcript, max_items=3))

            pain_points = _dedupe_preserve(pain_points, max_items=8)
            feature_requests = _dedupe_preserve(feature_requests, max_items=8)

            related_backlog_items: Dict[str, List[BacklogItemRef]] = {}
            for status, feats in self.backlog_by_area_status.get(area, {}).items():
                related_backlog_items[status] = [
                    BacklogItemRef(feature_id=f.feature_id, title=f.title, status=status, score=None, match_basis="area")
                    for f in feats
                ]

            matched = self._match_requests_to_backlog(feature_requests, top_n=5)

            gaps = []
            backlog_count = sum(len(v) for v in self.backlog_by_area_status.get(area, {}).values())
            if mentions_total >= 3 and backlog_count == 0:
                gaps.append("Research signal present but no backlog items exist for this product area.")
            if feature_requests and (not matched or (matched and (matched[0].score or 0.0) < 0.08)):
                gaps.append("Recurring feature requests have weak/no alignment with existing backlog items (possible roadmap gap).")

            reports.append(
                ProductAreaIntelligence(
                    product_area=area,
                    mentions_total=mentions_total,
                    mentions_by_source=mentions_by_source,
                    researcher_sentiment_distribution=dict(researcher_sent),
                    participant_sentiment_distribution=dict(participant_sent),
                    key_themes=themes,
                    pain_points=pain_points,
                    feature_requests=feature_requests,
                    related_backlog_items=related_backlog_items,
                    matched_backlog_from_requests=matched,
                    gaps=gaps,
                )
            )

        reports.sort(key=lambda r: r.mentions_total, reverse=True)
        return reports
