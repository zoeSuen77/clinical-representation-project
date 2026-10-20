"""Validate experiment settings and enumerate the required experiment matrix."""
import hashlib
import json
import math
import re

DEFAULT_METHODS = [
    ("tfidf", "lr"), ("tfidf", "svm"), ("bert", "lr"),
    ("biobert", "lr"), ("pubmedbert", "lr"),
]


def methods(config):
    pairs = [(m["features"], m["classifier"]) for m in config["methods"]] \
        if "methods" in config else list(DEFAULT_METHODS)
    if not pairs or len(pairs) != len(set(pairs)):
        raise ValueError("Methods must be nonempty and unique")
    for feature, classifier in pairs:
        if feature not in {"tfidf", "bert", "biobert", "pubmedbert"} \
                or classifier not in {"lr", "svm"}:
            raise ValueError(f"Unsupported method: {feature}/{classifier}")
    return pairs


def validate_config(config):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", config["name"]):
        raise ValueError("Experiment name must be a simple directory name")
    seeds = config["seeds"]
    if not seeds or len(seeds) != len(set(seeds)) or \
            any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Seeds must be unique nonnegative integers")
    budgets = config["budgets_per_class"]
    if not budgets or len(budgets) != len(set(budgets)) or any(
            b != "full" and (type(b) is not int or b <= 0) for b in budgets):
        raise ValueError("Budgets must be unique positive integers or 'full'")
    grid = config["C_grid"]
    if not grid or len(grid) != len(set(grid)) or any(
            type(c) not in {int, float} or not math.isfinite(c) or c <= 0 for c in grid):
        raise ValueError("C grid must contain unique finite positive values")
    for split, cap in config.get("split_caps", {}).items():
        if split not in {"train", "validation", "test"} or type(cap) is not int or cap <= 0:
            raise ValueError("Split caps must be positive integers for known splits")
    if type(config.get("full_pool_once", False)) is not bool:
        raise ValueError("full_pool_once must be boolean")
    for key in ("split_seed", "subset_seed"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if type(config["max_length"]) is not int or config["max_length"] < 3:
        raise ValueError("max_length must leave room for non-special tokens")
    if type(config["batch_size"]) is not int or config["batch_size"] < 1:
        raise ValueError("batch_size must be a positive integer")
    if config["pooling"] != "mean_non_special":
        raise ValueError("Only mean_non_special pooling is supported by this runner")
    methods(config)


def conditions(config):
    """Legacy configs repeat full; final configs may fit full only at seed 1."""
    validate_config(config)
    return [(seed, budget) for seed in config["seeds"]
            for budget in config["budgets_per_class"]
            if not (budget == "full" and config.get("full_pool_once", False)
                    and seed != config["seeds"][0])]


def expected_matrix(config):
    return {(f, c, seed, str(budget)) for f, c in methods(config)
            for seed, budget in conditions(config)}


def ids_sha256(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def config_sha256(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def check_model_registry(config, registry):
    for key, expected in config.get("model_revisions", {}).items():
        actual = {k: registry.get(key, {}).get(k) for k in ("repo_id", "revision")}
        if actual != expected:
            raise ValueError(f"Model registry differs from frozen protocol: {key}")
