"""Descriptive diagnostics only. No synthetic benchmark accuracy is computed."""
import argparse
import csv
import json
from pathlib import Path
import joblib
import numpy as np
from src.data_loader import save_json
from src.feature_extraction import extract_embeddings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pilot.json")
    p.add_argument("--features", nargs="+", default=["tfidf", "bert"])
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    demo = json.loads(Path("data/minimal_pairs_demo.json").read_text())
    rows = [{"id": f"demo:{pair['id']}:{side}", "text": pair[side]}
            for pair in demo["pairs"] for side in ["a", "b"]]
    all_results = {}
    for feature in args.features:
        model_path = Path(f"data/cache/{config['name']}_{feature}_lr.joblib")
        if not model_path.exists():
            raise FileNotFoundError(f"Run the full-pool LR experiment first: {model_path}")
        model = joblib.load(model_path)  # Only load local files produced by this project.
        if feature == "tfidf":
            X = np.array([r["text"] for r in rows])
            vectors = model["preprocessing"].transform(X).toarray()
        else:
            X, metadata = extract_embeddings(rows, feature, config, args.device)
            vectors = X
        norms = np.linalg.norm(vectors, axis=1)
        preds = model.predict(X)
        results = []
        for i, pair in enumerate(demo["pairs"]):
            a, b = 2*i, 2*i+1
            cosine = (float(vectors[a] @ vectors[b] / (norms[a] * norms[b]))
                      if norms[a] and norms[b] else None)
            results.append({**pair, "cosine_similarity": cosine,
                            "prediction_a": str(preds[a]), "prediction_b": str(preds[b]),
                            "predictions_agree": bool(preds[a] == preds[b])})
        all_results[feature] = results
    output = {
        "disclosure": demo["status"],
        "interpretation": "Predictions are cue-presence labels. A sentence-level model "
        "cannot assign separate statuses to different target diseases. Agreement is "
        "descriptive, not target-assertion accuracy. High cosine similarity alone "
        "does not show failure to encode negation.",
        "results": all_results}
    save_json(f"results/metrics/{config['name']}/minimal_pairs_demo.json", output)
    flat = [{"method": method, **p} for method, pairs in all_results.items() for p in pairs]
    columns = ["method", "id", "category", "relation", "a", "b", "cosine_similarity",
               "prediction_a", "prediction_b", "predictions_agree", "target"]
    with Path(f"results/metrics/{config['name']}/minimal_pairs_demo.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)
    notes = ("# Minimal-pair diagnostics\n\n" + demo["status"] + "\n\n"
             "All classifiers were fitted on the full pilot pool (seed 13). "
             "The examples were never used to train or tune models. Their clinical "
             "wording differs from the biomedical abstract training domain.\n\n"
             "| Method | Pair | Cosine | Prediction A | Prediction B |\n"
             "|---|---|---:|---|---|\n")
    for method, pairs in all_results.items():
        for pair in pairs[:2] + pairs[-2:]:
            cos = pair["cosine_similarity"]
            notes += (f"| {method} | {pair['id']} | "
                      f"{cos:.4f}" if cos is not None else f"| {method} | {pair['id']} | undefined")
            notes += f" | {pair['prediction_a']} | {pair['prediction_b']} |\n"
    notes += (
        "\nThe pneumonia/no-pneumonia pair demonstrates that a high cosine can coexist "
        "with different predicted cue labels. Geometry alone is not an assertion test.\n\n"
        "In `scope_1`, pneumonia and diabetes exchange roles while both sentences retain "
        "a negation cue. Equal sentence-level predictions are compatible with the trained "
        "task and cannot tell us whether the model resolved the target's status. In "
        "`scope_2`, an extra negation changes the construction; a cue-presence classifier "
        "cannot reliably adjudicate the resulting target meaning.\n\n"
        "Abbreviation and paraphrase pairs are included in the CSV/JSON. Agreement on a "
        "few authored pairs is a descriptive observation, not evidence of generalization "
        "to an independent medical benchmark. No synthetic benchmark accuracy is reported.\n")
    Path("results/minimal_pair_notes.md").write_text(notes)
    print(f"Saved {len(demo['pairs'])} demonstration pairs for {args.features}.")


if __name__ == "__main__":
    main()
