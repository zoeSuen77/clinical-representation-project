"""Generate figures and summary tables from saved raw experiment results."""
import os
os.environ.setdefault("MPLCONFIGDIR", str(__import__("pathlib").Path(".mplconfig").resolve()))
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize(name):
    base = Path("results/metrics") / name
    records, raw = [], []
    for path in sorted(base.glob("*_runs.json")):
        for run in json.loads(path.read_text()):
            raw.append(run)
            records.append({k: run[k] for k in
                            ["features", "classifier", "seed", "budget_per_class", "n_train"]} |
                           {k: run["metrics"][k] for k in
                            ["accuracy", "macro_f1", "macro_precision", "macro_recall"]})
    if not records:
        raise ValueError("No actual results to plot")
    frame = pd.DataFrame(records)
    frame.to_csv(base / "raw_metrics.csv", index=False)
    summary = frame.groupby(["features", "classifier", "budget_per_class"], sort=False).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"), precision_mean=("macro_precision", "mean"),
        recall_mean=("macro_recall", "mean"), n_train=("n_train", "first"),
        n_runs=("seed", "size")).reset_index()
    summary.to_csv(base / "summary.csv", index=False)
    dest = Path("results/figures") / name
    dest.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for (features, classifier), group in summary.groupby(["features", "classifier"]):
        group = group[group.budget_per_class != "full"].sort_values("n_train")
        ax.errorbar(group.n_train, group.macro_f1_mean, yerr=group.macro_f1_std.fillna(0),
                    marker="o", capsize=3, label=f"{features} + {classifier.upper()}")
    ax.set(xlabel="Labeled training sentences (balanced across 3 classes)",
           ylabel="Test macro-F1", ylim=(0, 1), title=f"BioScope cue detection: {name}")
    ax.grid(alpha=.18)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(dest / "learning_curves.png", dpi=180)
    fig.savefig(dest / "learning_curves.pdf")
    plt.close(fig)
    for run in raw:
        if run["budget_per_class"] != "full" or run["seed"] != 13:
            continue
        cm = np.array(run["metrics"]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.imshow(cm, cmap="Blues")
        labels = ["No cue", "Negation", "Speculation"]
        ax.set(xticks=range(3), yticks=range(3), xticklabels=labels, yticklabels=labels,
               xlabel="Predicted", ylabel="True",
               title=f"{run['features']} + {run['classifier'].upper()} ({name}, full pool)")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                        color="white" if cm[i,j] > cm.max()/2 else "black")
        fig.tight_layout()
        fig.savefig(dest / f"confusion_{run['features']}_{run['classifier']}.png", dpi=180)
        plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="pilot")
    summarize(p.parse_args().name)
