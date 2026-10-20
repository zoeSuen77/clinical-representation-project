"""Plan by default; explicitly execute or resume a complete audited experiment."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from filelock import FileLock
from experiments.audit_results import audit_results
from experiments.check_environment import check_environment
from experiments.preflight import preflight
from src.data_loader import save_json
from src.protocol import expected_matrix, methods, validate_config


def feature_groups(config):
    return list(dict.fromkeys(f for f, _ in methods(config)))


def run(config_path, device="cuda", resume=False):
    config = json.loads(Path(config_path).read_text())
    validate_config(config)
    base = Path("results/metrics") / config["name"]
    logs = Path("results/logs") / config["name"]
    logs.mkdir(parents=True, exist_ok=True)
    with FileLock(str(logs / "launch.lock"), timeout=0):
        environment = check_environment(device)
        save_json(logs / "environment.json", environment)
        if not environment["passed"]:
            raise ValueError(f"Environment check failed: {environment['failures']}")
        if config.get("expected_raw_sha256"):
            preflight(config)
        dataset = Path("data/processed") / f"{config['name']}.jsonl"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        status = {"state": "running", "experiment": config["name"], "device": device,
                  "started_utc": timestamp, "completed_features": [], "resumed_features": []}
        save_json(base / "launch_status.json", status)
        try:
            for feature in feature_groups(config):
                selected = [(f, c) for f, c in methods(config) if f == feature]
                if resume:
                    try:
                        audit_results(config, dataset, base, selected_methods=selected)
                        if feature != "tfidf":
                            meta = json.loads((base / f"{feature}_features.json").read_text())
                            if meta["device"] != device:
                                raise ValueError("Saved features use a different device")
                    except (ValueError, KeyError, TypeError, OSError):
                        pass
                    else:
                        status["resumed_features"].append(feature)
                        save_json(base / "launch_status.json", status)
                        print(f"Validated saved {feature}; skipping fitting", flush=True)
                        continue
                command = [sys.executable, "-m", "experiments.run_experiments",
                           "--config", str(config_path), "--features", feature, "--device", device]
                if feature != "tfidf" and (feature, "svm") in selected:
                    command.append("--embedding-svm")
                if resume:
                    command.append("--overwrite")
                log = logs / f"{timestamp}_{feature}.log"
                print(f"Running {feature}; log: {log}", flush=True)
                with log.open("w") as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
                audit_results(config, dataset, base, selected_methods=selected)
                status["completed_features"].append(feature)
                save_json(base / "launch_status.json", status)
            report = audit_results(config, dataset, base)
            save_json(base / "integrity_audit.json", report)
            subprocess.run([sys.executable, "-m", "src.visualization", "--name", config["name"]], check=True)
            status["state"] = "complete"
        except Exception as exc:
            status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
            save_json(base / "launch_status.json", status)
            raise
        save_json(base / "launch_status.json", status)
        return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/final_main.json")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--execute", action="store_true", help="Actually run fitting and evaluation")
    p.add_argument("--resume", action="store_true", help="Audit and skip complete feature groups; refit incomplete groups")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    validate_config(config)
    if not args.execute:
        print(json.dumps({"mode": "plan_only", "experiment": config["name"],
                          "device": args.device, "evaluations": len(expected_matrix(config)),
                          "feature_order": feature_groups(config), "training_started": False}, indent=2))
        return
    run(args.config, args.device, args.resume)


if __name__ == "__main__":
    main()
