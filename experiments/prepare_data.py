"""Download, parse, audit, and split the BioScope abstracts corpus."""
import argparse
import json
import urllib.request
from pathlib import Path
from src.data_loader import (parse_bioscope, clean_and_group, construct_splits,
                             validate_splits, save_json, write_jsonl, sha256_file)

URL = "https://rgai.inf.u-szeged.hu/sites/rgai.inf.u-szeged.hu/files/abstracts_pmid.xml"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pilot.json")
    p.add_argument("--xml", default="data/raw/abstracts_pmid.xml")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    path = Path(args.xml)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            path.write_bytes(response.read())
    raw = parse_bioscope(path)
    cleaned, audit = clean_and_group(raw)
    rows = construct_splits(cleaned, config)
    audit.update(source_url=URL, raw_sha256=sha256_file(path), config=config,
                 experiment_counts=validate_splits(rows),
                 task="Sentence-level annotated cue presence, not clinical assertion.")
    write_jsonl("data/processed/all_annotations.jsonl", raw)
    write_jsonl(f"data/processed/{config['name']}.jsonl", rows)
    # A manifest contains no article text and allows independent split auditing.
    save_json(f"data/processed/{config['name']}_manifest.json",
              [{k: r[k] for k in ["id", "doc_id", "group_id", "label", "split"]} for r in rows])
    save_json(f"results/metrics/{config['name']}_data_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
