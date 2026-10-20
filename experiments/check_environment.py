"""Check pinned dependencies and device operation without model training."""
import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from src.data_loader import save_json


def check_environment(device="cpu", lock="requirements-lock.txt"):
    import torch
    from packaging.version import Version
    versions, failures = {}, []
    for line in Path(lock).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        versions[name] = actual
        if actual == "missing" or Version(actual).public != Version(expected).public:
            failures.append(f"{name}: expected {expected}, found {actual}")
    if sys.version_info[:2] != (3, 12):
        failures.append("Use Python 3.12 for the frozen experiment")
    cuda = torch.cuda.is_available()
    if device == "cuda" and not cuda:
        failures.append("CUDA unavailable; install the compatible CUDA PyTorch wheel and check driver")
    device_test = False
    if device != "cuda" or cuda:
        try:
            x = torch.ones((8, 8), device=device)
            device_test = bool(torch.all(x @ x == 8).item())
        except Exception as exc:
            failures.append(f"Device operation failed: {type(exc).__name__}: {exc}")
    if not device_test and not failures:
        failures.append("Device arithmetic failed")
    free_gib = shutil.disk_usage(Path.cwd()).free / 1024**3
    if free_gib < 5:
        failures.append("Less than 5 GiB free in the project filesystem")
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    changed = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    return {"passed": not failures, "python": platform.python_version(),
            "source_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "source_has_local_changes": bool(changed.stdout.strip()) if changed.returncode == 0 else None,
            "platform": platform.system(), "architecture": platform.machine(),
            "requested_device": device, "device_test_passed": device_test,
            "cuda_available": cuda, "torch_cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
            "free_disk_gib": round(free_gib, 2), "packages": versions, "failures": failures,
            "scope": "Dependency versions and tensor operation; not a full experiment or cross-machine reproduction."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--output", default="results/pretrain/environment.json")
    args = p.parse_args()
    report = check_environment(args.device)
    save_json(args.output, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
