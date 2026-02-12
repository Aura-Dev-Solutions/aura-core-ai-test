# Technical Assessment

## User Research Intelligence Pipeline

### Context

**InsightFlow** is a B2B SaaS analytics company. Our product teams have been conducting user research interviews across the organization for the past year, but the resulting data is scattered across tools, formats, and researcher notebooks. Leadership wants to extract structured product intelligence from this unstructured research — not manually, but through a systematic pipeline.

You have been given real (anonymized) research artifacts from three sources. Your task is to build a pipeline that transforms messy, unstructured interview data and researcher notes into structured product intelligence.

---

## The Data

Three datasets are provided across two batches (`data/batch_1/` and `data/batch_2/`). Batch 1 is the initial load; batch 2 represents new research arriving later.

### Dataset A: Interview Transcripts (`interview_transcripts.json`)

Behavioral (BDD-style) user research interviews. Each interview is a JSON object containing metadata and a full transcript of the conversation.

| Field | Type | Description |
|---|---|---|
| `interview_id` | string | Unique identifier (`INT-NNN`) |
| `date` | string | When the interview was conducted |
| `participant_id` | string | Participant identifier (`P-NNN`) |
| `participant_role` | string | Participant's job role |
| `participant_company_size` | string | Size of participant's company |
| `interviewer` | string | Researcher who conducted the interview |
| `interview_type` | string | Type of interview (`behavioral`, `usability`, `discovery`, `feedback`) |
| `product_areas_discussed` | array | Product areas covered (when tagged by the researcher) |
| `transcript` | string | Full interview transcript text |
| `duration_minutes` | int | Interview length in minutes |
| `recording_quality` | string | Audio/transcript quality (`good`, `fair`, `poor`) |

**Batch 1:** ~40 interviews | **Batch 2:** ~12 interviews

### Dataset B: Research Session Notes (`session_notes.csv`)

Researcher notes taken during or after interviews and discovery sessions. Format and detail level vary significantly by researcher.

| Column | Type | Description |
|---|---|---|
| `note_id` | string | Unique identifier (`RN-NNNN`) |
| `created_at` | string | When the note was created |
| `researcher` | string | Author of the note |
| `related_interview_id` | string | Reference to an interview transcript (when available) |
| `participant_id` | string | Participant the note is about |
| `note_type` | string | Category (`observation`, `insight`, `pain_point`, `feature_request`, `quote`, `action_item`) |
| `content` | string | Note content |
| `product_area_tag` | string | Product area (when tagged) |
| `priority` | string | Importance level (`high`, `medium`, `low`) |
| `sentiment` | string | Observed sentiment (`positive`, `negative`, `neutral`, `mixed`) |

**Batch 1:** ~250 notes | **Batch 2:** ~80 notes

### Dataset C: Product Backlog (`product_backlog.json`)

The structured product backlog containing planned, in-progress, and shipped features.

| Field | Type | Description |
|---|---|---|
| `feature_id` | string | Unique identifier (`FEAT-NNN`) |
| `title` | string | Feature title |
| `description` | string | Feature description |
| `product_area` | string | Product area this feature belongs to |
| `status` | string | Current status (`planned`, `in_progress`, `shipped`, `deprecated`) |
| `priority` | string | Priority level (`p0`, `p1`, `p2`, `p3`) |
| `release_date` | string | Release date for shipped items (ISO-8601) |
| `tags` | array | Relevant tags |
| `quarter` | string | Target or actual quarter (`Q1-2024`, `Q2-2024`, etc.) |

**Batch 1:** ~80 items | **Batch 2:** ~15 items

### Data Quality

These are real-world research artifacts. They contain quality issues — inconsistent formats, missing values, varied transcription quality, and other problems you would expect from research data collected by multiple people over time. Discovering and handling these issues systematically is part of the assessment. Do not assume the data is clean.

---

## Expected Outcomes

Build a pipeline that produces the following. We care about **what** you deliver, not a specific implementation approach.

### 1. Unified Research Taxonomy

Classify every interview and research note into product areas. The result should be a consistent, structured mapping from each research artifact to one or more product areas — even when the original data uses informal or inconsistent language.

### 2. Product Area Intelligence Report

For each product area, produce a structured summary that includes:
- Volume of research mentions (interviews + notes)
- Sentiment distribution from researcher observations and participant language
- Key themes, pain points, and recurring feature requests
- Related items from the product backlog (shipped and planned)
- Gaps — areas with research signal but no corresponding backlog activity

### 3. Incremental Processing

When batch 2 arrives, your pipeline should process the new data **without reprocessing batch 1 from scratch**. Demonstrate that your architecture supports incremental updates.

### 4. Design Documentation (`DESIGN.md`)

A written document covering:
- Your data model and architecture decisions
- How you approached data quality issues (especially unstructured text)
- Your AI/ML approach — what you used, why, and what alternatives you considered
- Known limitations and what you would do differently with more time

---

## Guidelines

- **Language:** Python
- **AI approach:** Use any AI/ML technique you see fit — LLMs, embeddings, classical NLP, heuristics, or a combination. There is no single correct approach. Justify your choices.
- **Infrastructure:** No Docker, cloud deployment, or CI/CD required. We want to see your thinking, not your DevOps skills.
- **Reproducibility:** Your pipeline should be runnable with clear instructions.

### What We Are Looking For

- **Data modeling instincts** — how you structure messy, multi-source research data into something coherent
- **NLP and text processing skill** — extracting structure from unstructured interview transcripts and varied note formats
- **Systematic data quality handling** — not just fixing issues one by one, but building an approach
- **Thoughtful AI application** — choosing the right tool for the job, not the flashiest
- **Software design** — code that could evolve, not a one-off script
- **Clear communication** — your DESIGN.md matters as much as your code

### What We Are NOT Looking For

- Perfect accuracy on every classification
- Over-engineered infrastructure (Docker, microservices, etc.)
- A specific tech stack or framework
- Exhaustive test coverage — a few meaningful tests beat 100% coverage of trivial code

---

## Submission

Fork or branch this repository. Submit a pull request or share your repository link when complete.

**Questions?** Reach out via email to **otorres@auraresearch.ai** / **igutierrez@auraresearch.ai** or open an issue in this repository. Communication is valued — asking good questions is a positive signal, not a weakness.
