"""Fail closed on missing, stale, inconsistent, or corrupted experiment results."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from src.data_loader import LABELS, read_jsonl, validate_splits, save_json, sha256_file, training_indices
from src.evaluation import evaluate
from src.protocol import methods, conditions, expected_matrix, ids_sha256, config_sha256, validate_config


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_metrics(saved, expected, location="metrics"):
    if isinstance(expected, dict):
        require(isinstance(saved, dict) and set(saved) == set(expected), f"Metric keys disagree: {location}")
        for key, value in expected.items():
            check_metrics(saved[key], value, f"{location}.{key}")
    elif isinstance(expected, list):
        require(isinstance(saved, list) and len(saved) == len(expected), f"Metric list length disagrees: {location}")
        for i, (left, right) in enumerate(zip(saved, expected)):
            check_metrics(left, right, f"{location}[{i}]")
    elif isinstance(expected, (int, float)):
        require(type(saved) in {int, float} and math.isfinite(saved)
                and np.isclose(saved, expected, rtol=1e-8, atol=1e-10), f"Recomputed metric disagrees: {location}")
    else:
        require(saved == expected, f"Metric value disagrees: {location}")


def audit_results(config, dataset_path, base, *, selected_methods=None):
    validate_config(config)
    dataset_path, base = Path(dataset_path), Path(base)
    rows = read_jsonl(dataset_path)
    counts = validate_splits(rows)
    by_id = {r["id"]: r for r in rows}
    test_ids = {r["id"] for r in rows if r["split"] == "test"}
    protocol = json.loads((base / "protocol.json").read_text())
    require(protocol["config"] == config, "Saved protocol differs from requested config")
    require(protocol["dataset_sha256"] == sha256_file(dataset_path), "Dataset checksum differs from saved protocol")
    require(protocol["test_n"] == len(test_ids), "Protocol test count disagrees")
    require(protocol["validation_n"] == sum(r["split"] == "validation" for r in rows), "Protocol validation count disagrees")
    if config.get("protocol_version"):
        require(protocol.get("config_sha256") == config_sha256(config), "Config checksum differs from saved protocol")
    checked_methods = methods(config) if selected_methods is None else list(selected_methods)
    require(checked_methods and len(checked_methods) == len(set(checked_methods))
            and set(checked_methods) <= set(methods(config)), "Invalid audit method selection")
    expected = {k for k in expected_matrix(config) if k[:2] in checked_methods}
    expected_files = {f"{f}_{c}_runs.json" for f, c in checked_methods}
    actual_files = {p.name for p in base.glob("*_runs.json")}
    require(actual_files == expected_files if selected_methods is None else expected_files <= actual_files,
            f"Method files incomplete or unexpected; missing={sorted(expected_files-actual_files)}, extra={sorted(actual_files-expected_files)}")
    subsets = {(s, str(b)): [rows[i]["id"] for i in training_indices(rows, b, s)] for s, b in conditions(config)}
    seen, warnings, method_names = set(), [], []
    for features, classifier in checked_methods:
        prefix = f"{features}_{classifier}"
        method_names.append(prefix)
        require(json.loads((base / f"{prefix}_skipped.json").read_text()) == [], f"Skipped planned experiments for {prefix}")
        runs = json.loads((base / f"{prefix}_runs.json").read_text())
        require(isinstance(runs, list), f"Expected run list: {prefix}")
        for run in runs:
            require(run["features"] == features and run["classifier"] == classifier, f"Run identity disagrees with filename: {prefix}")
            key = (features, classifier, run["seed"], str(run["budget_per_class"]))
            require(key in expected and key not in seen, f"Unexpected or duplicate run: {key}")
            seen.add(key)
            ids = subsets[(run["seed"], str(run["budget_per_class"]))]
            require(run["train_ids"] == ids and run["n_train"] == len(ids), f"Training IDs differ from configured sampling: {key}")
            require(run["train_ids_sha256"] == ids_sha256(ids), f"Training checksum disagrees: {key}")
            stem = f"{prefix}_s{run['seed']}_n{run['budget_per_class']}"
            require(json.loads((base / f"{stem}.json").read_text()) == run, f"Individual run and aggregate disagree: {stem}")
            predictions = json.loads((base / f"{stem}_predictions.json").read_text())
            require(len(predictions) == len(test_ids) and {p["id"] for p in predictions} == test_ids,
                    f"Missing or duplicate test prediction IDs: {stem}")
            require(all(p["true"] == by_id[p["id"]]["label"] and p["pred"] in LABELS for p in predictions), f"Invalid prediction labels: {stem}")
            check_metrics(run["metrics"], evaluate([p["true"] for p in predictions], [p["pred"] for p in predictions]))
            trials = run["validation_trials"]
            require([t["C"] for t in trials] == config["C_grid"], f"Incomplete or reordered C search: {stem}")
            require(all(type(t["validation_macro_f1"]) in {int, float} and math.isfinite(t["validation_macro_f1"])
                        and 0 <= t["validation_macro_f1"] <= 1 for t in trials), f"Invalid validation score: {stem}")
            require(max(trials, key=lambda r: r["validation_macro_f1"])["C"] == run["C"], f"Selected C violates validation selection: {stem}")
            require(isinstance(run["convergence_warnings"], list), "Invalid warning record")
            warnings.extend(run["convergence_warnings"])
        if features != "tfidf":
            meta = json.loads((base / f"{features}_features.json").read_text())
            require(meta["n_rows"] == len(rows) and meta["model_frozen"] is True, f"Feature count or frozen status disagrees: {features}")
            spec = config.get("model_revisions", {}).get(features)
            if spec:
                require(all(meta["model"].get(k) == v for k, v in spec.items()), f"Model revision differs from protocol: {features}")
    require(seen == expected, f"Incomplete experiment matrix: {len(seen)}/{len(expected)}")
    require(not warnings, f"Convergence warnings require resolution: {len(warnings)}")
    return {"passed": True, "experiment": config["name"], "complete_protocol": selected_methods is None,
            "expected_evaluations": len(expected),
            "evaluations_checked": len(seen), "methods": method_names, "shared_training_subsets": len(subsets),
            "split_counts": counts, "convergence_warning_count": len(warnings),
            "config_sha256": config_sha256(config), "dataset_sha256": sha256_file(dataset_path),
            "checks": ["complete method and seed/budget matrix", "document/group/exact-text split isolation",
                       "training IDs match configured sampling and hashes", "individual and aggregate run records agree",
                       "one prediction per test ID with valid labels", "all metrics and confusion cells recomputed",
                       "complete C grid and first-best validation selection", "frozen encoders and pinned revisions when specified"],
            "limits": "Saved records cannot independently prove validation score provenance or that test results never influenced design. Source review and independent reruns remain necessary."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--output", help="Optional report path without changing saved results")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    validate_config(config)
    base = Path("results/metrics") / config["name"]
    output = Path(args.output) if args.output else base / "integrity_audit.json"
    try:
        report = audit_results(config, Path("data/processed") / f"{config['name']}.jsonl", base)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        save_json(output, {"passed": False, "experiment": config["name"], "error": str(exc)})
        parser.exit(1, f"Audit FAILED: {exc}\n")
    save_json(output, report)
    print(f"Verified {report['evaluations_checked']}/{report['expected_evaluations']} evaluations for {config['name']}.")


if __name__ == "__main__":
    main()
