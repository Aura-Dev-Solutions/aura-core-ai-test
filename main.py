import os
import json
import argparse
from typing import List

from src.ingestor import ResearchIngestor
from src.processor import ResearchProcessor
from src.models import SessionNote, Interview
from src.utils import sanitize_text


def _sanitize_more(text: str) -> str:
    """
    Extra sanitization without touching utils.py.
    """
    t = sanitize_text(text or "")
    # remove additional headers that may appear in notes
    t = (
        t.replace("## Pain Point", "")
         .replace("## Feature Request", "")
         .replace("## Observation", "")
         .replace("## Quote", "")
    )
    return " ".join((t or "").split()).strip()


def _dump_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _append_to_artifact_store(store_path: str, notes, interviews) -> None:
    os.makedirs(os.path.dirname(store_path), exist_ok=True)

    existing_keys = set()
    if os.path.exists(store_path):
        with open(store_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing_keys.add((obj.get("artifact_type"), obj.get("artifact_id")))
                except Exception:
                    continue

    with open(store_path, "a", encoding="utf-8") as f:
        for n in notes:
            key = ("note", n.note_id)
            if key in existing_keys:
                continue
            row = {"artifact_type": "note", "artifact_id": n.note_id, "data": n.model_dump()}
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

        for iv in interviews:
            key = ("interview", iv.interview_id)
            if key in existing_keys:
                continue
            row = {"artifact_type": "interview", "artifact_id": iv.interview_id, "data": iv.model_dump()}
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")



def _load_artifact_store(store_path: str):
    notes, interviews = [], []
    if not os.path.exists(store_path):
        return notes, interviews

    with open(store_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj["artifact_type"] == "note":
                notes.append(SessionNote(**obj["data"]))
            else:
                interviews.append(Interview(**obj["data"]))
    return notes, interviews



def _write_taxonomy_mapping_jsonl(
    output_path: str,
    batch_number: int,
    notes,
    interviews,
) -> None:
    """
    Outcome #1: structured mapping from each artifact to 1+ product areas.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # Notes
        for n in notes:
            areas = [a.model_dump() for a in (n.taxonomy or [])]
            primary = areas[0]["product_area"] if areas else "unclassified"
            row = {
                "artifact_type": "note",
                "artifact_id": n.note_id,
                "batch": batch_number,
                "primary_area": primary,
                "areas": areas,
                "source": "session_notes.csv",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Interviews
        for iv in interviews:
            areas = [a.model_dump() for a in (iv.taxonomy or [])]
            primary = areas[0]["product_area"] if areas else "unclassified"
            row = {
                "artifact_type": "interview",
                "artifact_id": iv.interview_id,
                "batch": batch_number,
                "primary_area": primary,
                "areas": areas,
                "source": "interview_transcripts.json",
                "participant_sentiment": iv.participant_sentiment,
                "participant_sentiment_score": iv.participant_sentiment_score,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_pipeline(batch_number: int, data_root: str, output_dir: str, state_file: str) -> None:
    print(f"\n--- Starting Pipeline: Batch {batch_number} ---")

    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    ingestor = ResearchIngestor(state_file=state_file)

    # 1) Paths
    backlog_path = os.path.join(data_root, f"batch_{batch_number}", "product_backlog.json")
    note_path = os.path.join(data_root, f"batch_{batch_number}", "session_notes.csv")
    interview_path = os.path.join(data_root, f"batch_{batch_number}", "interview_transcripts.json")

    # 2) Initialize components
    ingestor = ResearchIngestor(state_file=state_file)
    backlog_data = ingestor.ingest_backlog(backlog_path)
    processor = ResearchProcessor(backlog=backlog_data)

    # 3) Ingest new data (incremental behavior controlled by state_file)
    raw_notes = ingestor.ingest_notes(note_path)
    raw_interviews = ingestor.ingest_interviews(interview_path)

    if not raw_notes and not raw_interviews:
        print(f"No new data to process for Batch {batch_number}.")
        return

    # 4) Sanitization + batch tagging
    for note in raw_notes:
        note.content = _sanitize_more(note.content)
        note.source_batch = batch_number

    for iv in raw_interviews:
        iv.transcript = _sanitize_more(iv.transcript)

    # 5) Taxonomy
    unified_notes = processor.unify_taxonomy(raw_notes)
    classified_interviews = processor.classify_interviews(raw_interviews)

    # Persist processed artifacts for cumulative reporting (no batch1 reprocessing)
    artifact_store = os.path.join(output_dir, "artifact_store.jsonl")
    _append_to_artifact_store(artifact_store, unified_notes, classified_interviews)

    # Cumulative report (uses store only)
    all_notes, all_interviews = _load_artifact_store(artifact_store)
    cumulative_report = processor.generate_intelligence_report(all_notes, all_interviews)

    cumulative_path = os.path.join(output_dir, f"intelligence_report_cumulative_up_to_batch_{batch_number}.json")
    _dump_json(cumulative_path, [r.model_dump() for r in cumulative_report])
    print(f"✅ Cumulative report generated: {cumulative_path}")


    # 6) Intelligence Report (Outcome #2)
    report = processor.generate_intelligence_report(unified_notes, classified_interviews)

    # 7) Outputs
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, f"intelligence_report_batch_{batch_number}.json")
    _dump_json(report_path, [r.model_dump() for r in report])

    taxonomy_path = os.path.join(output_dir, f"taxonomy_mapping_batch_{batch_number}.jsonl")
    _write_taxonomy_mapping_jsonl(
        taxonomy_path,
        batch_number=batch_number,
        notes=unified_notes,
        interviews=classified_interviews,
    )

    print(f"✅ Report generated: {report_path}")
    print(f"✅ Taxonomy mapping generated: {taxonomy_path}")
    print(f"Processed {len(unified_notes)} notes and {len(classified_interviews)} interviews.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data", help="Root folder containing batch_1/, batch_2/, ...")
    ap.add_argument("--output-dir", default="output", help="Where outputs are written")
    ap.add_argument("--batch", type=int, default=0, help="Run a single batch number (e.g., 1). If 0, runs 1 then 2.")
    ap.add_argument("--reset-state", action="store_true", help="Delete pipeline state file (forces reprocessing).")
    args = ap.parse_args()

    # Put state in output_dir so it's not accidentally committed and doesn't break the reviewer run
    state_file = os.path.join(args.output_dir, "pipeline_state.json")
    if args.reset_state and os.path.exists(state_file):
        os.remove(state_file)
        print(f"State reset: removed {state_file}")

    if args.batch and args.batch > 0:
        run_pipeline(args.batch, args.data_root, args.output_dir, state_file)
    else:
        run_pipeline(1, args.data_root, args.output_dir, state_file)
        run_pipeline(2, args.data_root, args.output_dir, state_file)


if __name__ == "__main__":
    main()
