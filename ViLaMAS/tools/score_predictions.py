"""Score already generated multiple-choice predictions without model inference."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.benchmark_io import (
    InputError, emit_report, index_records, load_annotations, load_labels, read_records,
)


def summarize(ids: list[str], correct: dict[str, bool], missing: set[str],
              invalid: set[str]) -> dict:
    n = len(ids)
    hits = sum(correct[sample_id] for sample_id in ids)
    return {
        "questions": n, "correct": hits,
        "accuracy_percent": round(100.0 * hits / n, 6) if n else None,
        "missing_predictions": sum(sample_id in missing for sample_id in ids),
        "invalid_predictions": sum(sample_id in invalid for sample_id in ids),
    }


def score(annotations: dict[str, dict], labels: dict[str, int],
          predictions: dict[str, dict]) -> dict:
    if set(predictions) - set(annotations):
        raise InputError("predictions: contains ids outside the selected evaluation set.")
    missing = set(annotations) - set(predictions)
    invalid = set()
    correct = {}
    for sample_id, row in annotations.items():
        answer = predictions.get(sample_id, {}).get("prediction")
        valid = type(answer) is int and 0 <= answer < len(row["options"])
        if sample_id not in missing and not valid:
            invalid.add(sample_id)
        correct[sample_id] = valid and answer == labels[sample_id]
    grouped = {}
    for group in ("short", "medium", "long", "unassigned"):
        ids = [
            sample_id for sample_id, row in annotations.items()
            if (row.get("duration_group") or "unassigned") == group
        ]
        if ids:
            grouped[group] = summarize(ids, correct, missing, invalid)
    return {
        "tool": "score_predictions",
        "scope": "the supplied annotation set",
        "answer_encoding": "zero-based integer option index",
        "missing_and_invalid_policy": "count as incorrect; retain in denominator",
        "overall": summarize(list(annotations), correct, missing, invalid),
        "by_duration": grouped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="Normalized JSON or JSONL records.")
    parser.add_argument("--labels", required=True, help="JSON object: question id -> option index.")
    parser.add_argument("--predictions", required=True, help="JSON array or JSONL objects: id, prediction.")
    parser.add_argument("--output", help="Create a JSON report; existing files are not overwritten.")
    args = parser.parse_args()
    try:
        annotations = load_annotations(args.annotations)
        labels = load_labels(args.labels, annotations)
        predictions = index_records(read_records(args.predictions, "predictions"), "predictions")
        emit_report(score(annotations, labels, predictions), args.output)
        return 0
    except InputError as error:
        emit_report({"tool": "score_predictions", "passed": False, "error": str(error)}, None)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
