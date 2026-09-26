"""Report package/CUDA availability without loading weights or printing host paths."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from importlib import metadata
import io
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.benchmark_io import InputError, emit_report

PACKAGES = (
    "torch", "torchvision", "transformers", "accelerate", "qwen-vl-utils",
    "decord", "av", "numpy", "pillow", "pandas", "pyarrow", "huggingface-hub",
    "safetensors", "PyYAML", "psutil", "nvidia-ml-py",
)


def inspect_environment(require_gpu: bool) -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    missing = [name for name, version in versions.items() if version is None]
    report = {
        "tool": "check_env", "python": platform.python_version(),
        "python_supported": sys.version_info >= (3, 10),
        "packages": versions, "missing_packages": missing,
        "torch_import_ok": False, "qwen3_vl_import_ok": False,
        "cuda": {"available": False, "runtime_version": None, "devices": []},
        "require_gpu": require_gpu, "import_errors": {},
    }
    # Import messages can contain local paths. Report only exception class names.
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        if versions["torch"] is not None:
            try:
                import torch
                report["torch_import_ok"] = True
                report["cuda"]["runtime_version"] = torch.version.cuda
                report["cuda"]["available"] = torch.cuda.is_available()
                if report["cuda"]["available"]:
                    for index in range(torch.cuda.device_count()):
                        properties = torch.cuda.get_device_properties(index)
                        report["cuda"]["devices"].append({
                            "index": index, "name": properties.name,
                            "total_memory_gib": round(properties.total_memory / 1024**3, 2),
                        })
            except Exception as error:
                report["import_errors"]["torch"] = type(error).__name__
        if versions["transformers"] is not None:
            try:
                from transformers import Qwen3VLForConditionalGeneration
                report["qwen3_vl_import_ok"] = Qwen3VLForConditionalGeneration is not None
            except Exception as error:
                report["import_errors"]["transformers"] = type(error).__name__
    report["passed"] = (
        report["python_supported"] and not missing
        and report["torch_import_ok"] and report["qwen3_vl_import_ok"]
        and not report["import_errors"]
        and (not require_gpu or bool(report["cuda"]["devices"]))
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true", help="Fail if no CUDA GPU is available.")
    parser.add_argument("--output", help="Create a JSON report; existing files are not overwritten.")
    args = parser.parse_args()
    try:
        report = inspect_environment(args.require_gpu)
        emit_report(report, args.output)
        return 0 if report["passed"] else 1
    except InputError as error:
        emit_report({"tool": "check_env", "passed": False, "error": str(error)}, None)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
