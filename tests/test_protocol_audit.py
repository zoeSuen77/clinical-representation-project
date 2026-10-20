"""Exercise audit failure modes and a real TF-IDF run in an empty workspace."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from experiments.audit_results import audit_results
from src.data_loader import LABELS, save_json, write_jsonl, validate_splits
from src.duplicate_audit import group_near_duplicates
from src.protocol import conditions, expected_matrix, check_model_registry

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("clean-experiment")
    config = {
        "name": "smoke", "protocol_version": "test", "split_seed": 1729,
        "subset_seed": 2026, "split_caps": {}, "seeds": [13, 37],
        "budgets_per_class": [2, "full"], "full_pool_once": True,
        "C_grid": [0.1, 1.0], "max_length": 192, "batch_size": 16,
        "pooling": "mean_non_special",
        "methods": [{"features": "tfidf", "classifier": c} for c in ["lr", "svm"]],
    }
    phrases = ["observations remained stable", "no evidence absent", "possibly may suggest"]
    rows = []
    for split, n in [("train", 6), ("validation", 2), ("test", 2)]:
        for label, phrase in zip(LABELS, phrases):
            for i in range(n):
                key = f"{split}_{label}_{i}"
                rows.append({"id": key, "doc_id": key, "group_id": key,
                             "text": f"{phrase} unique_{key}", "label": label, "split": split})
    write_jsonl(root / "data/processed/smoke.jsonl", rows)
    save_json(root / "configs/smoke.json", config)
    save_json(root / "configs/models.json", {})
    command = [sys.executable, "-m", "experiments.run_experiments",
               "--config", "configs/smoke.json", "--features", "tfidf"]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    # The directory did not exist before training: a fresh checkout must work.
    assert (root / "data/cache/smoke_tfidf_lr.joblib").is_file()
    assert (root / "data/cache/smoke_tfidf_svm.joblib").is_file()
    return root, config, command, env


def test_empty_workspace_run_and_rerun_guard(completed_run):
    root, config, command, env = completed_run
    report = audit_results(config, root / "data/processed/smoke.jsonl", root / "results/metrics/smoke")
    assert report["passed"] and report["evaluations_checked"] == 6
    assert report["shared_training_subsets"] == 3
    receipt = root / "results/metrics/smoke/integrity_audit.json"
    save_json(receipt, report)
    refused = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    assert refused.returncode != 0 and "--overwrite" in refused.stderr
    assert json.loads(receipt.read_text())["passed"] is True
    rerun = subprocess.run(command + ["--overwrite"], cwd=root, env=env, capture_output=True, text=True)
    assert rerun.returncode == 0, rerun.stderr
    assert json.loads(receipt.read_text())["passed"] is False
    assert audit_results(config, root / "data/processed/smoke.jsonl",
                         root / "results/metrics/smoke")["passed"]


@pytest.mark.parametrize("damage,expected", [
    ("empty", "Method files incomplete"),
    ("missing_budget", "Incomplete experiment matrix"),
    ("duplicate_run", "Unexpected or duplicate run"),
    ("duplicate_prediction", "Missing or duplicate test prediction"),
    ("wrong_metric", "Recomputed metric disagrees"),
    ("incomplete_grid", "Incomplete or reordered C search"),
    ("wrong_training_ids", "Training IDs differ"),
    ("wrong_config", "Saved protocol differs"),
])
def test_audit_rejects_corrupt_or_incomplete_results(completed_run, tmp_path, damage, expected):
    root, config, _, _ = completed_run
    base = tmp_path / "results"
    shutil.copytree(root / "results/metrics/smoke", base)
    aggregate = base / "tfidf_lr_runs.json"
    runs = json.loads(aggregate.read_text())
    run = runs[0]
    stem = f"tfidf_lr_s{run['seed']}_n{run['budget_per_class']}"
    if damage == "empty":
        for p in base.glob("*_runs.json"):
            p.unlink()
    elif damage == "missing_budget":
        runs.pop()
        save_json(aggregate, runs)
    elif damage == "duplicate_run":
        runs.append(copy.deepcopy(run))
        save_json(aggregate, runs)
    elif damage == "duplicate_prediction":
        path = base / f"{stem}_predictions.json"
        pred = json.loads(path.read_text())
        pred[1] = pred[0]
        save_json(path, pred)
    elif damage == "wrong_config":
        protocol = json.loads((base / "protocol.json").read_text())
        protocol["config"]["split_seed"] += 1
        save_json(base / "protocol.json", protocol)
    else:
        if damage == "wrong_metric":
            run["metrics"]["per_class"]["no_cue"]["recall"] = 0.123456
        elif damage == "incomplete_grid":
            run["validation_trials"].pop()
        elif damage == "wrong_training_ids":
            run["train_ids"].reverse()
        save_json(base / f"{stem}.json", run)
        save_json(aggregate, runs)
    with pytest.raises(ValueError, match=expected):
        audit_results(config, root / "data/processed/smoke.jsonl", base)


def test_final_schedule_and_legacy_pilot_schedule():
    final = json.loads((ROOT / "configs/final_main.json").read_text())
    pilot = json.loads((ROOT / "configs/pilot.json").read_text())
    assert len(expected_matrix(final)) == 65
    assert [(s, b) for s, b in conditions(final) if b == "full"] == [(13, "full")]
    assert len(expected_matrix(pilot)) == 75


def test_near_duplicates_link_documents_before_splitting():
    text = " ".join(f"word{i}" for i in range(60))
    rows = [{"id": "a", "doc_id": "a", "group_id": "a", "text": text + " alpha"},
            {"id": "b", "doc_id": "b", "group_id": "b", "text": text + " beta"},
            {"id": "c", "doc_id": "c", "group_id": "c", "text": "unrelated short phrase"}]
    policy = {"method": "token_shingle_jaccard", "shingle_size": 5,
              "min_tokens": 8, "threshold": 0.9}
    grouped, report = group_near_duplicates(rows, policy)
    assert grouped[0]["group_id"] == grouped[1]["group_id"] != grouped[2]["group_id"]
    assert report["matching_pairs"] == 1 and report["sentences_below_min_tokens"] == 1
    assert group_near_duplicates(rows, policy) == (grouped, report)


def test_document_leakage_even_when_group_ids_are_wrong():
    rows = [{"id": f"{s}-{l}", "doc_id": f"{s}-{l}", "group_id": f"{s}-{l}",
             "text": f"text {s} {l}", "label": l, "split": s}
            for s in ["train", "validation", "test"] for l in LABELS]
    rows[-1]["doc_id"] = rows[0]["doc_id"]
    with pytest.raises(ValueError, match="Cross-split leakage"):
        validate_splits(rows)


def test_model_revision_change_is_rejected():
    config = {"model_revisions": {"bert": {"repo_id": "test/model", "revision": "pinned"}}}
    with pytest.raises(ValueError, match="Model registry differs"):
        check_model_registry(config, {"bert": {"repo_id": "test/model", "revision": "changed"}})
