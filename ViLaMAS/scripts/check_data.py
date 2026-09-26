"""Check normalized question records and, optionally, local media assets."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.benchmark_io import InputError, emit_report, load_annotations, relative_asset_path


def check_media(annotations: dict[str, dict], root: Path) -> dict[str, int]:
    root = root.resolve()
    if not root.is_dir():
        raise InputError("The data root must be an existing directory.")
    missing = {"missing_video_files": 0, "missing_subtitle_files": 0}
    for row in annotations.values():
        for field, metric in (
            ("video_path", "missing_video_files"),
            ("subtitle_path", "missing_subtitle_files"),
        ):
            value = row.get(field)
            if value is None:
                if field == "video_path":
                    missing[metric] += 1
                continue
            candidate = (root / relative_asset_path(value)).resolve()
            if not candidate.is_relative_to(root):
                raise InputError("An asset resolves outside the supplied data root.")
            if not candidate.is_file():
                missing[metric] += 1
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="Normalized JSON or JSONL records.")
    parser.add_argument("--check-files", action="store_true", help="Also check media file existence.")
    parser.add_argument("--data-root", type=Path, help="Local root for relative video/subtitle paths.")
    parser.add_argument("--output", help="Create a JSON report; existing files are not overwritten.")
    args = parser.parse_args()
    if args.check_files and args.data_root is None:
        parser.error("--check-files requires --data-root")
    try:
        annotations = load_annotations(args.annotations)
        rows = list(annotations.values())
        groups = {
            group: sum(row.get("duration_group") == group for row in rows)
            for group in ("short", "medium", "long")
        }
        groups["unassigned"] = sum(row.get("duration_group") is None for row in rows)
        report = {
            "tool": "check_data", "passed": True, "questions": len(rows),
            "duration_groups": groups,
            "records_with_video_path": sum(row.get("video_path") is not None for row in rows),
            "records_with_subtitle_path": sum(row.get("subtitle_path") is not None for row in rows),
            "media_files_checked": args.check_files,
        }
        if args.check_files:
            report.update(check_media(annotations, args.data_root))
            report["passed"] = (
                report["missing_video_files"] == report["missing_subtitle_files"] == 0
            )
        emit_report(report, args.output)
        return 0 if report["passed"] else 1
    except InputError as error:
        emit_report({"tool": "check_data", "passed": False, "error": str(error)}, None)
        return 2
    except (OSError, RuntimeError):
        emit_report({"tool": "check_data", "passed": False, "error": "Cannot inspect media files."}, None)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
