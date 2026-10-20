"""Read actual BioScope XML annotations and construct a cue-presence task.

This is not disease assertion classification. XML tags supply labels only and
are never included in the model input. Character spans retain source offsets.
"""
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ["no_cue", "negation_only", "speculation_only"]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def flatten_sentence(element):
    pieces, spans = [], []
    cursor = 0

    def walk(node):
        nonlocal cursor
        start = cursor
        if node.text:
            pieces.append(node.text)
            cursor += len(node.text)
        for child in node:
            walk(child)
            if child.tail:
                pieces.append(child.tail)
                cursor += len(child.tail)
        if node.tag in {"cue", "xcope"}:
            spans.append({"tag": node.tag, "start": start, "end": cursor, **node.attrib})

    walk(element)
    text = "".join(pieces)
    for span in spans:
        span["text"] = text[span["start"]:span["end"]]
    return text, spans


def parse_bioscope(path):
    root = ET.parse(path).getroot()
    rows = []
    for doc in root.iter("Document"):
        pmid = doc.findtext("DocID")
        if not pmid:
            raise ValueError("Missing document ID")
        for part in doc.findall("DocumentPart"):
            for sentence in part.iter("sentence"):
                text, spans = flatten_sentence(sentence)
                cue_types = {s["type"] for s in spans if s["tag"] == "cue"}
                if not cue_types <= {"negation", "speculation"}:
                    raise ValueError(f"Unknown cue types: {cue_types}")
                label = ("both" if len(cue_types) == 2 else
                         "negation_only" if "negation" in cue_types else
                         "speculation_only" if "speculation" in cue_types else "no_cue")
                rows.append({"id": f"{pmid}:{sentence.attrib['id']}",
                             "doc_id": pmid, "part": part.get("type"),
                             "text": text, "label": label, "spans": spans})
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Empty dataset or duplicate source IDs")
    return rows


def clean_and_group(rows):
    """Link documents sharing exact normalized sentences BEFORE deduplication."""
    parent = {r["doc_id"]: r["doc_id"] for r in rows}

    def find(a):
        if parent[a] != a:
            parent[a] = find(parent[a])
        return parent[a]

    def union(a, b):
        ra, rb = find(a), find(b)
        parent[max(ra, rb)] = min(ra, rb)

    by_text = defaultdict(list)
    for r in rows:
        by_text[normalize_text(r["text"])].append(r)
    for matches in by_text.values():
        for r in matches[1:]:
            union(matches[0]["doc_id"], r["doc_id"])
    cleaned, conflicting, duplicates = [], 0, 0
    for text, matches in by_text.items():
        if len({r["label"] for r in matches}) > 1:
            conflicting += len(matches)
            continue
        duplicates += len(matches) - 1
        r = dict(matches[0])
        if r["label"] == "both" or not text:
            continue
        r["group_id"] = find(r["doc_id"])
        cleaned.append(r)
    audit = {"raw_sentences": len(rows), "source_documents": len(parent),
             "raw_labels": dict(Counter(r["label"] for r in rows)),
             "conflicting_duplicate_rows_removed": conflicting,
             "redundant_duplicate_rows_removed": duplicates,
             "mixed_label_policy": "Exclude all both-cue sentences from the three-class task.",
             "retained_sentences": len(cleaned),
             "retained_labels": dict(Counter(r["label"] for r in cleaned)),
             "retained_document_groups": len({r["group_id"] for r in cleaned})}
    return cleaned, audit


def construct_splits(rows, config):
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split

    y = np.array([r["label"] for r in rows])
    groups = np.array([r["group_id"] for r in rows])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                   random_state=config["split_seed"])
    fold_id = np.empty(len(rows), dtype=int)
    for fold, (_, idx) in enumerate(splitter.split(np.zeros(len(rows)), y, groups)):
        fold_id[idx] = fold
    names = np.where(fold_id == 0, "test", np.where(fold_id == 1, "validation", "train"))
    selected = []
    for name in ["train", "validation", "test"]:
        idx = np.flatnonzero(names == name)
        cap = config.get("split_caps", {}).get(name)
        if cap and len(idx) > cap:
            idx, _ = train_test_split(idx, train_size=cap, stratify=y[idx],
                                      random_state=config["subset_seed"])
        for i in sorted(idx):
            selected.append({**rows[i], "split": name})
    validate_splits(selected)
    return selected


def validate_splits(rows):
    seen_doc, seen_text, seen_id = {}, {}, set()
    counts = defaultdict(Counter)
    for r in rows:
        if r["id"] in seen_id:
            raise ValueError("Duplicate ID in experiment data")
        seen_id.add(r["id"])
        for registry, value in [(seen_doc, r["group_id"]),
                                 (seen_text, normalize_text(r["text"]))]:
            if value in registry and registry[value] != r["split"]:
                raise ValueError(f"Cross-split leakage: {value}")
            registry[value] = r["split"]
        counts[r["split"]][r["label"]] += 1
    for name in ["train", "validation", "test"]:
        if set(counts[name]) != set(LABELS):
            raise ValueError(f"{name} does not contain all classes")
    return dict(counts)


def training_indices(rows, budget, seed):
    """Nested per-class subsets for a seed; the exact IDs are shared by models."""
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    if budget == "full":
        return idx
    picked = []
    for label in LABELS:
        pool = np.array([i for i in idx if rows[i]["label"] == label])
        if len(pool) < budget:
            raise ValueError(f"Budget {budget} exceeds {label} pool of {len(pool)}")
        picked.extend(rng.permutation(pool)[:budget].tolist())
    return np.array(sorted(picked))
