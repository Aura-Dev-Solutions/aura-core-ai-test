User Research Intelligence Pipeline

## 1. Overview

This project builds a Python pipeline that ingests real-world research artifacts (interview transcripts + session notes) alongside a structured product backlog. 

The pipeline produces:

1. **Unified Research Taxonomy**: maps **every** research artifact (note and interview) to **one or more product areas** with confidence and evidence.
2. **Product Area Intelligence Report**: for each product area, generates a structured summary with volume, sentiment (researcher vs. participant), themes, pain points, feature requests, backlog linkage (shipped/planned), and gaps.
3. **Incremental Processing**: processes `batch_2` without reprocessing `batch_1`, using persisted state.

> The goal is not perfect classification accuracy, but a robust, reproducible, and extensible system.

---

## 2. Inputs

* `data/batch_1/interview_transcripts.json` and `data/batch_2/interview_transcripts.json`
* `data/batch_1/session_notes.csv` and `data/batch_2/session_notes.csv`
* `data/batch_1/product_backlog.json` and `data/batch_2/product_backlog.json`

---

## 3. Outputs

### 3.1 Unified Research Taxonomy (Outcome #1)

* `output/taxonomy_mapping_batch_{N}.jsonl`
  Each line represents a research artifact with:
* `artifact_type`: `note` | `interview`
* `artifact_id`
* `primary_area`
* `areas[]`: list of `{product_area, confidence, method, evidence[]}`
* (for interviews) `participant_sentiment` and `participant_sentiment_score`

### 3.2 Product Area Intelligence Report (Outcome #2)

* `output/intelligence_report_batch_{N}.json`
  A list of per-`product_area` objects containing:
* `mentions_total`
* `mentions_by_source` (notes vs. interviews)
* `researcher_sentiment_distribution` (from notes)
* `participant_sentiment_distribution` (from interviews)
* `key_themes`
* `pain_points` and `feature_requests`
* `related_backlog_items` (split by status: planned/in_progress/shipped/deprecated/etc.)
* `matched_backlog_from_requests` (token overlap matching)
* `gaps`

In addition to per-batch (delta) reports, the pipeline also produces a cumulative report after each batch by aggregating from a persisted artifact store (no reprocessing of prior batch inputs).

- output/intelligence_report_batch_{N}.json  (delta: only artifacts newly processed in batch N)
- output/intelligence_report_cumulative_up_to_batch_{N}.json  (cumulative: batch_1..batch_N)


---

## 4. Data Model

The system treats notes and interviews as “research artifacts” enriched with taxonomy and NLP signals.

### 4.1 Research Artifacts

* **SessionNote**: `note_id`, `created_at`, `note_type`, `content`, `product_area_tag`, `sentiment`, etc.
* **Interview**: `interview_id`, `date`, `participant_id`, `transcript`, etc.

### 4.2 Taxonomy assignment

Each artifact contains:

* `taxonomy: List[ProductAreaAssignment]`
  with:
* `product_area`
* `confidence`
* `method` (currently: `keyword_scoring`)
* `evidence[]` (tokens/phrases supporting the decision)

### 4.3 Intelligence report schema

Delivered as `ProductAreaIntelligence` (see `models.py`), aligned 1:1 with the requested rubric.

---

## 5. Architecture / Flow

Pipeline per batch:

1. **Ingest**

   * Load backlog, notes, interviews.
   * Sanitize text (remove headers like “## Insight”, “## Quote”, etc.).
2. **Normalize / Data Quality**

   * Normalize `note_type` (typos) and `sentiment`.
   * Normalize `product_area` (synonyms, casing, hyphens).
   * Defensive defaults for missing fields (pipeline does not crash).
3. **Unified taxonomy**

   * `unify_taxonomy(notes)` classifies notes.
   * `classify_interviews(interviews)` classifies interviews.
4. **Enrichment & Report**

   * Extract themes (n-grams) from combined text.
   * Extract pain points / requests with rule-based patterns.
   * Link backlog by:

     * direct match on `product_area`
     * additional match from requests (token overlap)
   * Detect gaps with heuristics.
5. **Write outputs**

   * Taxonomy mapping `.jsonl`
   * Intelligence report `.json`

---

## 6. AI approach

The approach is intentionally **simple, interpretable, and reproducible**.

### 6.1 Taxonomy classification

* **Keyword scoring using a lexicon per product area**:

  * Lexicon built from backlog (title + tags) + a small manual phrase list.
  * Synonym normalization (`billing/payments → billing`, `dataimport → data-import`, etc.)
* Output: multi-label taxonomy with “confidence” derived from relative scoring.


### 6.2 Sentiment

* **Researcher sentiment**: read from the `sentiment` field in notes when available.
* **Participant sentiment**: estimated from transcripts using a small positive/negative lexicon to produce distributions by area.

### 6.3 Themes / pain points / feature requests

* Themes: frequent n-grams with stopword filtering.
* Pain points / requests: sentence-level extraction using patterns (e.g., “can’t”, “frustrating”, “wish”, “please add”).

---

## 7. Data quality handling

Artifacts are messy (inconsistent formats, missing fields, text headers, etc.). The strategy is:

* Systematic text sanitization.
* Centralized normalization for key fields.
* Defensive handling: when fields are missing, apply safe defaults and continue.
* Classification evidence: each label includes phrases that justify the assignment.

> This avoids one-off “manual fixes” and keeps the logic evolvable.

---

## 8. Incremental processing

The pipeline supports incremental updates across batches.

### File-level ingestion state
A persisted state file is maintained at:
- `output/pipeline_state.json`

This state tracks which input batch files have already been ingested so that when `batch_2` arrives, the pipeline does not re-ingest or reprocess `batch_1` inputs.

A `--reset-state` flag is provided to clear this state for reproducible reruns. Also a `--batch 1` or `--batch 2` functionality is added in case new data was added to any batch or manual loading wants to be done.

### Cumulative reporting without reprocessing prior batches
To align with the expectation that the system “updates” when new research arrives, the pipeline also produces a cumulative intelligence report after each batch.

Processed artifacts (notes + interviews) are persisted to:
- `output/artifact_store.jsonl`

Each stored record contains the sanitized text plus enrichment results (taxonomy labels and sentiment). After processing batch N, the pipeline generates:
- a delta report from newly processed artifacts in batch N, and
- a cumulative report by aggregating over `artifact_store.jsonl` (covering batch_1..batch_N).

This produces cumulative results without reprocessing batch_1 raw inputs when batch_2 arrives.

---

## 9. Reproducibility

Run:

```bash
python main.py --reset-state
python main.py
```

This processes batch 1 and then batch 2, producing:

* `output/taxonomy_mapping_batch_1.jsonl`
* `output/intelligence_report_batch_1.json`
* `output/intelligence_report_cumulative_up_to_batch_1.json`
* `output/taxonomy_mapping_batch_2.jsonl`
* `output/intelligence_report_batch_2.json`
* `output/intelligence_report_cumulative_up_to_batch_2.json`

---

## 10. Known limitations / Future work

* As global research grows, hard-coded regex for multiple languages becomes unmaintainable.
* Keyword-based classification can miss implicit language.
* Transcript sentiment mixes interviewer questions and participant answers; speaker-aware parsing would improve accuracy.
* Record-level incremental updates instead of file-level.


---
