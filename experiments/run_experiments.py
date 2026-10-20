"""Examples: python -m experiments.run_experiments --features tfidf bert."""
import argparse
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from src.data_loader import read_jsonl, save_json, sha256_file, training_indices, validate_splits
from src.classifiers import tune_classifier
from src.evaluation import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pilot.json")
    p.add_argument("--features", nargs="+", default=["tfidf"],
                   choices=["tfidf", "bert", "biobert", "pubmedbert"])
    p.add_argument("--embedding-svm", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--extract-only", action="store_true")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    name = config["name"]
    dataset = Path(f"data/processed/{name}.jsonl")
    rows = read_jsonl(dataset)
    validate_splits(rows)
    y = np.array([r["label"] for r in rows])
    val = np.array([i for i, r in enumerate(rows) if r["split"] == "validation"])
    test = np.array([i for i, r in enumerate(rows) if r["split"] == "test"])
    base = Path("results/metrics") / name
    base.mkdir(parents=True, exist_ok=True)
    metadata = {"config": config, "dataset_sha256": sha256_file(dataset),
                "python": platform.python_version(), "sklearn": sklearn.__version__,
                "numpy": np.__version__, "test_n": len(test), "validation_n": len(val)}
    save_json(base / "protocol.json", metadata)
    for features in args.features:
        if features == "tfidf":
            X = np.array([r["text"] for r in rows])
        else:
            from src.feature_extraction import extract_embeddings
            X, feature_metadata = extract_embeddings(rows, features, config, args.device)
            save_json(base / f"{features}_features.json", feature_metadata)
        if args.extract_only:
            continue
        classifiers = ["lr", "svm"] if features == "tfidf" or args.embedding_svm else ["lr"]
        for classifier in classifiers:
            results, skipped = [], []
            for seed in config["seeds"]:
                for budget in config["budgets_per_class"]:
                    try:
                        train = training_indices(rows, budget, seed)
                    except ValueError as exc:
                        skipped.append({"seed": seed, "budget": budget, "reason": str(exc)})
                        continue
                    start = time.perf_counter()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        model, trials = tune_classifier(
                            X[train], y[train], X[val], y[val], features, classifier,
                            config["C_grid"], seed)
                    predictions = model.predict(X[test])
                    ids = [rows[i]["id"] for i in train]
                    result = {"features": features, "classifier": classifier, "seed": seed,
                              "budget_per_class": budget, "n_train": len(train),
                              "train_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                              "train_ids": ids, "C": model["classifier"].C,
                              "validation_trials": trials, "metrics": evaluate(y[test], predictions),
                              "seconds": time.perf_counter() - start,
                              "convergence_warnings": [str(w.message) for w in caught]}
                    results.append(result)
                    run_name = f"{features}_{classifier}_s{seed}_n{budget}"
                    save_json(base / f"{run_name}.json", result)
                    # Test predictions contain IDs and labels, not licensed article text.
                    save_json(base / f"{run_name}_predictions.json",
                              [{"id": rows[i]["id"], "true": str(y[i]), "pred": str(pred)}
                               for i, pred in zip(test, predictions)])
                    if budget == "full" and seed == config["seeds"][0]:
                        model_path = Path("data/cache") / f"{name}_{features}_{classifier}.joblib"
                        joblib.dump(model, model_path)
                    print(f"{name} {features}/{classifier} seed={seed} budget={budget}: "
                          f"F1={result['metrics']['macro_f1']:.4f}", flush=True)
            save_json(base / f"{features}_{classifier}_runs.json", results)
            save_json(base / f"{features}_{classifier}_skipped.json", skipped)


if __name__ == "__main__":
    main()
