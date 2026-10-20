"""Technical end-to-end check on 30 synthetic sentences; no BioScope training."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from src.data_loader import LABELS, save_json, write_jsonl, sha256_file


def synthetic_rows():
    phrases = ["The sample contains the measured protein",
               "No evidence of the measured protein was found",
               "The measured protein may possibly be present"]
    rows = []
    for split, n in [("train", 6), ("validation", 2), ("test", 2)]:
        for label, phrase in zip(LABELS, phrases):
            for i in range(n):
                key = f"synthetic-{split}-{label}-{i}"
                rows.append({"id": key, "doc_id": key, "group_id": key, "label": label,
                             "text": f"{phrase} in {split} observation number {i}.", "split": split})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--workdir", required=True, help="New, empty smoke workspace")
    p.add_argument("--output", default="results/pretrain/smoke.json")
    p.add_argument("--offline", action="store_true", help="Require already downloaded checkpoint files")
    args = p.parse_args()
    root, output = Path.cwd(), Path(args.output).resolve()
    workdir = Path(args.workdir).resolve()
    if workdir.exists():
        raise FileExistsError("Use a new smoke directory to prove cold feature-cache creation")
    workdir.mkdir(parents=True)
    os.environ.setdefault("HF_HOME", str(root / "data/cache/huggingface"))
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PYTHONPATH"] = str(root)
    config = json.loads((root / "configs/final_main.json").read_text())
    for key in ("expected_raw_sha256", "near_duplicates"):
        config.pop(key, None)
    config.update(name="synthetic_smoke", split_caps={}, seeds=[13], budgets_per_class=[2, "full"],
                  protocol_version="synthetic-technical-check", note="Not a scientific benchmark")
    save_json(workdir / "configs/smoke.json", config)
    save_json(workdir / "configs/models.json", json.loads((root / "configs/models.json").read_text()))
    shutil.copy2(root / "requirements-lock.txt", workdir / "requirements-lock.txt")
    rows = synthetic_rows()
    write_jsonl(workdir / "data/processed/synthetic_smoke.jsonl", rows)
    command = [sys.executable, "-m", "experiments.launch_final", "--config", "configs/smoke.json",
               "--device", args.device, "--execute"]
    report = {"passed": False, "started_utc": datetime.now(timezone.utc).isoformat(),
              "device": args.device, "scope": "30 authored synthetic sentences; no BioScope model training."}
    save_json(output, report)
    subprocess.run(command, cwd=workdir, check=True)
    os.chdir(workdir)
    from src.feature_extraction import extract_embeddings
    cache_checks = []
    for key in ("bert", "biobert", "pubmedbert"):
        features = json.loads((Path("results/metrics/synthetic_smoke") / f"{key}_features.json").read_text())
        cache = Path(features["cache_path"])
        before = sha256_file(cache)
        X, metadata = extract_embeddings(rows, key, config, args.device)
        if not metadata["cache_hit"] or X.shape != (30, 768) or not np.isfinite(X).all() \
                or sha256_file(cache) != before:
            raise ValueError(f"Cache round-trip failed: {key}")
        cache_checks.append({"model": key, "shape": list(X.shape), "cache_hit": True,
                             "unchanged_checksum": True, "model_frozen": metadata["model_frozen"],
                             "revision": metadata["model"]["revision"], "device": metadata["device"]})
    metrics = Path("results/metrics/synthetic_smoke")
    before = {p.name: sha256_file(p) for p in metrics.glob("*_runs.json")}
    subprocess.run(command + ["--resume"], check=True)
    after = {p.name: sha256_file(p) for p in metrics.glob("*_runs.json")}
    if before != after:
        raise ValueError("Resume changed already audited run records")
    audit = json.loads((metrics / "integrity_audit.json").read_text())
    status = json.loads((metrics / "launch_status.json").read_text())
    if len(status["resumed_features"]) != 4:
        raise ValueError("Resume failed to skip all four validated feature groups")
    report.update(passed=True, synthetic_evaluations=audit["evaluations_checked"],
                  cache_checks=cache_checks, resume_skipped_features=status["resumed_features"],
                  resume_preserved_results=True, generated_summary=(metrics / "summary.csv").exists(),
                  generated_figures=len(list(Path("results/figures/synthetic_smoke").glob("*.png"))),
                  final_bioscope_evaluations=0)
    save_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
