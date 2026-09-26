"""Standard-library readers for the documented, normalized preview format."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any


class InputError(ValueError):
    """An input error with a message that does not expose filesystem paths."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError("JSON objects must not contain duplicate keys.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise InputError("JSON must not contain non-finite numbers.")


def _decode(text: str, description: str) -> Any:
    try:
        return json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except json.JSONDecodeError:
        raise InputError(f"{description}: invalid JSON syntax.") from None


def read_json(path: str | Path, description: str) -> Any:
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        raise InputError(f"{description}: cannot read a UTF-8 file.") from None
    return _decode(text, description)


def read_records(path: str | Path, description: str) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        try:
            with path.open(encoding="utf-8-sig") as stream:
                rows = [
                    _decode(line, f"{description}, line {number}")
                    for number, line in enumerate(stream, 1) if line.strip()
                ]
        except (OSError, UnicodeError):
            raise InputError(f"{description}: cannot read a UTF-8 file.") from None
    else:
        rows = read_json(path, description)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise InputError(f"{description}: expected a JSON array or JSONL objects.")
    return rows


def index_records(rows: list[dict[str, Any]], description: str) -> dict[str, dict]:
    indexed = {}
    for number, row in enumerate(rows, 1):
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise InputError(f"{description}, record {number}: id must be a nonempty string.")
        if sample_id in indexed:
            raise InputError(f"{description}, record {number}: duplicate id.")
        indexed[sample_id] = row
    return indexed


def relative_asset_path(value: str) -> PurePosixPath:
    """Accept portable local paths; exclude URLs, drive letters and traversal."""
    path = PurePosixPath(value)
    if (not value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value)
            or "\\" in value or ":" in value
            or path.is_absolute() or ".." in path.parts or str(path) == "."):
        raise InputError("Asset paths must be relative POSIX paths inside the data root.")
    return path


def load_annotations(path: str | Path) -> dict[str, dict]:
    rows = read_records(path, "annotations")
    if not rows:
        raise InputError("annotations: the evaluation set must be nonempty.")
    indexed = index_records(rows, "annotations")
    for number, row in enumerate(rows, 1):
        prefix = f"annotations, record {number}"
        if any(field in row for field in ("answer", "gold", "label", "labels", "correct")):
            raise InputError(f"{prefix}: keep ground-truth answers in a separate labels file.")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise InputError(f"{prefix}: question must be a nonempty string.")
        options = row.get("options")
        if (not isinstance(options, list) or len(options) < 2
                or any(not isinstance(option, str) or not option.strip() for option in options)):
            raise InputError(f"{prefix}: options must contain at least two nonempty strings.")
        if row.get("duration_group") not in (None, "short", "medium", "long"):
            raise InputError(f"{prefix}: duration_group must be short, medium, long, or null.")
        for field in ("video_path", "subtitle_path"):
            value = row.get(field)
            if value is not None:
                if not isinstance(value, str):
                    raise InputError(f"{prefix}: asset paths must be strings or null.")
                relative_asset_path(value)
    return indexed


def load_labels(path: str | Path, annotations: dict[str, dict]) -> dict[str, int]:
    labels = read_json(path, "labels")
    if not isinstance(labels, dict):
        raise InputError("labels: expected an object mapping question ids to option indices.")
    if set(labels) != set(annotations):
        raise InputError("labels: ids must exactly match the selected annotation set.")
    for sample_id, answer in labels.items():
        if type(answer) is not int or not 0 <= answer < len(annotations[sample_id]["options"]):
            raise InputError("labels: each answer must be a valid zero-based integer index.")
    return labels


def emit_report(report: dict, output: str | None) -> None:
    """Print JSON; optionally create a new report without overwriting a file."""
    text = json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False) + "\n"
    if output is not None:
        try:
            destination = Path(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
        except FileExistsError:
            raise InputError("Output already exists; choose a new report filename.") from None
        except OSError:
            raise InputError("Cannot write the output report.") from None
    print(text, end="")
