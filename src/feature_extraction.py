"""Frozen Transformer features with immutable model revisions and content caching."""
import hashlib
import json
import os
import random
import time
from pathlib import Path
import numpy as np
from src.data_loader import save_json

os.environ.setdefault("HF_HOME", str(Path("data/cache/huggingface").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def masked_mean(hidden, attention_mask, special_tokens_mask):
    weights = (attention_mask.bool() & ~special_tokens_mask.bool()).unsqueeze(-1)
    if (weights.sum(1) == 0).any():
        raise ValueError("No non-special tokens available for pooling")
    return (hidden * weights).sum(1) / weights.sum(1)


def target_span_mean(hidden, offsets, attention_mask, spans):
    """Optional character-span pooling. Fail on missing or truncated targets."""
    import torch
    vectors = []
    for i, (start, end) in enumerate(spans):
        if start < 0 or end <= start:
            raise ValueError("Invalid target span")
        mask = ((offsets[i, :, 0] < end) & (offsets[i, :, 1] > start) &
                attention_mask[i].bool())
        selected = offsets[i][mask]
        if not len(selected) or selected[0, 0] > start or selected[-1, 1] < end:
            raise ValueError("Target span absent or truncated")
        vectors.append(hidden[i][mask].mean(0))
    return torch.stack(vectors)


def resolve_spec(key, registry_path="configs/models.json"):
    from huggingface_hub import HfApi
    registry = json.loads(Path(registry_path).read_text())
    spec = registry[key]
    if not spec.get("revision"):
        spec["revision"] = HfApi().model_info(spec["repo_id"]).sha
        save_json(registry_path, registry)
    return spec


def extract_embeddings(rows, key, config, device="auto"):
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModel

    spec = resolve_spec(key)
    payload = {"model": spec, "max_length": config["max_length"],
               "pooling": "last_hidden_state; mean over attended, non-special tokens",
               "transformers": transformers.__version__, "torch": torch.__version__,
               "rows": [(r["id"], r["text"]) for r in rows]}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    dest = Path("data/cache") / f"{key}_{fingerprint[:20]}.npz"
    metadata_path = dest.with_suffix(".json")
    if dest.exists() and metadata_path.exists():
        cached = np.load(dest, allow_pickle=False)
        if cached["ids"].tolist() != [r["id"] for r in rows]:
            raise ValueError("Cached feature IDs disagree")
        return cached["embeddings"], json.loads(metadata_path.read_text())

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(spec["repo_id"], revision=spec["revision"])
    model = AutoModel.from_pretrained(spec["repo_id"], revision=spec["revision"])
    model.requires_grad_(False)
    model.eval().to(device)
    texts = [r["text"] for r in rows]
    lengths = [len(tokenizer(t, truncation=False)["input_ids"]) for t in texts]
    order = np.argsort(lengths, kind="stable")
    embeddings = np.empty((len(rows), model.config.hidden_size), dtype=np.float32)
    started = time.perf_counter()
    for start in range(0, len(order), config["batch_size"]):
        idx = order[start:start + config["batch_size"]]
        batch = tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                          max_length=config["max_length"], return_tensors="pt",
                          return_special_tokens_mask=True)
        special = batch.pop("special_tokens_mask").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
            pooled = masked_mean(hidden, batch["attention_mask"], special)
        embeddings[idx] = pooled.cpu().numpy()
        if start % (config["batch_size"] * 20) == 0:
            print(f"{key}: {min(start + len(idx),len(rows))}/{len(rows)}", flush=True)
    if not np.isfinite(embeddings).all():
        raise ValueError("Non-finite embeddings")
    metadata = {k: v for k, v in payload.items() if k != "rows"}
    metadata.update(fingerprint=fingerprint, cache_path=str(dest), device=device,
                    n_rows=len(rows), dimensions=int(embeddings.shape[1]),
                    truncated_sentences=sum(n > config["max_length"] for n in lengths),
                    seconds=time.perf_counter() - started,
                    model_frozen=all(not p.requires_grad for p in model.parameters()))
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, embeddings=embeddings, ids=np.array([r["id"] for r in rows]))
    save_json(metadata_path, metadata)
    return embeddings, metadata
