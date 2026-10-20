"""Verify prepared data and planned training subsets without fitting a model."""
import argparse
import json
from pathlib import Path
from src.data_loader import (read_jsonl, validate_splits, training_indices,
                             save_json, sha256_file)
from src.protocol import (methods, conditions, expected_matrix, ids_sha256,
                          config_sha256, check_model_registry, validate_config)


def prepare_training_plan(config, rows):
    counts = validate_splits(rows)
    entries = []
    for seed, budget in conditions(config):
        idx = training_indices(rows, budget, seed)
        ids = [rows[i]["id"] for i in idx]
        entries.append({"seed": seed, "budget_per_class": budget,
                        "n_train": len(ids), "train_ids_sha256": ids_sha256(ids),
                        "train_ids": ids})
    return counts, entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/final_main.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    validate_config(config)
    dataset = Path("data/processed") / f"{config['name']}.jsonl"
    rows = read_jsonl(dataset)
    counts, entries = prepare_training_plan(config, rows)
    audit = json.loads((Path("results/metrics") / f"{config['name']}_data_audit.json").read_text())
    if audit["config"] != config:
        raise ValueError("Prepared data audit does not match requested config")
    if audit["prepared_dataset_sha256"] != sha256_file(dataset):
        raise ValueError("Prepared dataset checksum differs from data audit")
    if audit["raw_sha256"] != config.get("expected_raw_sha256", audit["raw_sha256"]):
        raise ValueError("Raw dataset hash differs from protocol")
    registry = json.loads(Path("configs/models.json").read_text())
    check_model_registry(config, registry)
    manifest_path = Path("data/processed") / f"{config['name']}_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest != [{k: r[k] for k in ("id", "doc_id", "group_id", "label", "split")} for r in rows]:
        raise ValueError("Prepared split manifest does not match data")
    for split, cap in config.get("split_caps", {}).items():
        if sum(counts[split].values()) != cap:
            raise ValueError(f"{split} does not have the configured cap of {cap}")
    base = Path("results/metrics") / config["name"]
    save_json(base / "training_plan.json",
              {"config_sha256": config_sha256(config),
               "dataset_sha256": sha256_file(dataset), "subsets": entries})
    report = {
        "status": "ready_for_training", "experiment": config["name"],
        "config_sha256": config_sha256(config), "dataset_sha256": sha256_file(dataset),
        "manifest_sha256": sha256_file(manifest_path), "split_counts": counts,
        "split_document_counts": audit["split_document_counts"],
        "split_group_counts": audit["split_group_counts"],
        "retained_sentences": audit["retained_sentences"], "selected_sentences": len(rows),
        "excluded_by_split_caps": audit["retained_sentences"] - len(rows),
        "planned_methods": [f"{f}/{c}" for f, c in methods(config)],
        "planned_evaluations": len(expected_matrix(config)),
        "planned_training_subsets": len(entries),
        "model_revisions": config.get("model_revisions", {}),
        "near_duplicate_matching_pairs": audit.get("near_duplicates", {}).get("matching_pairs"),
        "checks": ["all classes present", "no document/group/exact-text cross-split leakage",
                   "budgets feasible", "split manifest agrees with prepared data",
                   "validation size matches config", "model registry matches frozen protocol"],
        "claim": "Data and sampling preflight only; no final classifier was fitted or evaluated.",
    }
    save_json(base / "preflight.json", report)
    print(json.dumps({k: report[k] for k in ("status", "split_counts", "planned_evaluations")},
                     indent=2))


if __name__ == "__main__":
    main()
