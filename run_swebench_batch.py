"""Run a JSON array of SWE-bench instances serially with resumable output."""

import argparse
import fcntl
import hashlib
import json
import tempfile
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

from minimal_agent.models import OpenAIModel
from run_swebench_instance import run_instance


def _fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _write_atomic(path: Path, content: str) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def _read_artifact(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_prediction(path: Path, instance_id: str) -> dict:
    prediction = _read_artifact(path)
    patch = prediction.get("model_patch")
    if prediction.get("instance_id") == instance_id and isinstance(patch, str) and patch.strip():
        return prediction
    return {}


def _finish_attempt(
    attempt: dict, output_dir: Path, instance_id: str, *,
    elapsed: float | None = None, error: Exception | None = None,
) -> None:
    run_dir = output_dir / attempt["run_directory"]
    trajectory = _read_artifact(run_dir / "trajectory.json")
    info = trajectory.get("info", {})
    prediction = _read_prediction(run_dir / "predictions.jsonl", instance_id)
    status = "interrupted" if elapsed is None else "failed"
    if prediction:
        status = "submitted"
    attempt.update(
        status=status,
        elapsed_seconds=round(elapsed, 3) if elapsed is not None else None,
        exit_status=info.get("exit_status") or None,
        model_stats=info.get("model_stats"),
    )
    if error is not None:
        attempt["error_type"] = type(error).__name__
        detail = getattr(error, "stderr", None) or str(error)
        attempt["error_message"] = str(detail).strip()[-1000:]


def _save_results(output_dir: Path, state: dict) -> None:
    predictions = []
    for instance_id, attempts in state["instances"].items():
        latest = attempts[-1]
        if latest["status"] == "submitted":
            path = output_dir / latest["run_directory"] / "predictions.jsonl"
            prediction = _read_prediction(path, instance_id)
            if not prediction:
                raise ValueError(f"Missing or invalid saved prediction for {instance_id}: {path}")
            predictions.append(json.dumps(prediction, ensure_ascii=False))
    _write_atomic(output_dir / "results.json", json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    _write_atomic(output_dir / "predictions.jsonl", "".join(line + "\n" for line in predictions))


def run_batch(instances: list[dict], config: dict, output_dir: Path, *, retry_failed: bool = False) -> dict:
    if not isinstance(instances, list) or not instances or any(
        not isinstance(instance, dict)
        or not isinstance(instance.get("instance_id"), str)
        or not instance["instance_id"]
        for instance in instances
    ):
        raise ValueError("The manifest must be a nonempty JSON array of instances with instance IDs")
    instance_ids = [instance["instance_id"] for instance in instances]
    if len(set(instance_ids)) != len(instance_ids):
        raise ValueError("The manifest contains duplicate instance IDs")

    metadata = {
        "instance_ids": instance_ids,
        "manifest_sha256": _fingerprint(instances),
        "config_sha256": _fingerprint(config),
        "model": config["model"]["model"],
        "step_limit": config["agent"].get("step_limit", 0),
        "cost_limit": config["agent"].get("cost_limit", 3.0),
    }
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another batch is using {output_dir}") from None

        results_path = output_dir / "results.json"
        state = {"run": metadata, "instances": {}}
        if results_path.exists():
            state = json.loads(results_path.read_text(encoding="utf-8"))
            if state["run"] != metadata:
                raise ValueError("This output directory belongs to a different manifest or configuration")
        _save_results(output_dir, state)

        model = None
        for index, instance in enumerate(instances, start=1):
            instance_id = instance["instance_id"]
            attempts = state["instances"].setdefault(instance_id, [])
            if attempts:
                latest = attempts[-1]
                if latest["status"] == "running":
                    _finish_attempt(latest, output_dir, instance_id)
                    _save_results(output_dir, state)
                if latest["status"] == "submitted" or (latest["status"] == "failed" and not retry_failed):
                    print(f"[{index}/{len(instances)}] {instance_id}: skipped ({latest['status']})", flush=True)
                    continue

            if model is None:
                model = OpenAIModel(**config["model"])
            run_dir = Path(tempfile.mkdtemp(prefix=f"{index:03d}-", dir=output_dir))
            attempt = {"status": "running", "run_directory": str(run_dir.relative_to(output_dir))}
            attempts.append(attempt)
            _save_results(output_dir, state)

            started = time.monotonic()
            error = None
            try:
                run_instance(instance, model, config["agent"], output_dir, run_dir=run_dir)
            except Exception as exc:
                error = exc
            _finish_attempt(attempt, output_dir, instance_id, elapsed=time.monotonic() - started, error=error)
            _save_results(output_dir, state)
            result = attempt["exit_status"] or attempt.get("error_type") or attempt["status"]
            print(f"[{index}/{len(instances)}] {instance_id}: {attempt['status']} ({result})", flush=True)
    return state


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON array of SWE-bench dataset rows")
    parser.add_argument("--output-dir", type=Path, required=True, help="Batch output and resume directory")
    parser.add_argument("--config", type=Path, default=root / "minimal_agent/config/default.yaml")
    parser.add_argument("--step-limit", type=int, help="Override the per-instance model call limit")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed instances in an existing batch")
    args = parser.parse_args()
    if args.step_limit is not None and args.step_limit < 0:
        parser.error("--step-limit must be nonnegative")

    load_dotenv(root / ".env")
    instances = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.step_limit is not None:
        config["agent"]["step_limit"] = args.step_limit
    run_batch(instances, config, args.output_dir, retry_failed=args.retry_failed)


if __name__ == "__main__":
    main()
