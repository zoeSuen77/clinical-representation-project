"""Check saved results against predictions and shared training manifests."""
import json
from pathlib import Path
import numpy as np
from src.data_loader import read_jsonl, validate_splits, save_json
from src.evaluation import evaluate


def main():
    rows = read_jsonl("data/processed/pilot.jsonl")
    counts = validate_splits(rows)
    by_id = {r["id"]:r for r in rows}
    test_ids = {r["id"] for r in rows if r["split"] == "test"}
    shared = {}
    total, warnings = 0, []
    methods = []
    for path in sorted(Path("results/metrics/pilot").glob("*_runs.json")):
        methods.append(path.stem.removesuffix("_runs"))
        for run in json.loads(path.read_text()):
            total += 1
            key = (run["seed"], str(run["budget_per_class"]))
            ids = run["train_ids"]
            assert len(ids) == len(set(ids)) == run["n_train"]
            assert all(by_id[i]["split"] == "train" for i in ids)
            if key in shared:
                assert shared[key] == ids, f"Different labeled examples at {key}"
            shared[key] = ids
            stem = f"{run['features']}_{run['classifier']}_s{run['seed']}_n{run['budget_per_class']}"
            pred = json.loads((path.parent / f"{stem}_predictions.json").read_text())
            assert {r["id"] for r in pred} == test_ids
            assert all(r["true"] == by_id[r["id"]]["label"] for r in pred)
            recomputed = evaluate([r["true"] for r in pred], [r["pred"] for r in pred])
            for k in ["accuracy", "macro_f1", "macro_precision", "macro_recall"]:
                assert np.isclose(recomputed[k], run["metrics"][k])
            best = max(run["validation_trials"], key=lambda r:r["validation_macro_f1"])
            assert best["C"] == run["C"]
            warnings.extend(run["convergence_warnings"])
    save_json("results/metrics/pilot/integrity_audit.json",
              {"passed":True, "evaluations_checked":total, "methods":methods,
               "shared_training_subsets":len(shared), "split_counts":counts,
               "convergence_warning_count":len(warnings),
               "checks":["document-group and exact-duplicate isolation",
                         "identical training IDs across all methods",
                         "test predictions match fixed test IDs",
                         "metrics recomputed from predictions",
                         "C selected only by validation macro-F1"]})
    print(f"Verified {total} evaluations across {len(methods)} methods. "
          f"Convergence warnings: {len(warnings)}")


if __name__ == "__main__":
    main()
