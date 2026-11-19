"""Run one SWE-bench instance from a dataset row saved as JSON."""

import argparse
import json
import re
import tempfile
from pathlib import Path

import yaml
from dotenv import load_dotenv

from minimal_agent.agent import DefaultAgent
from minimal_agent.environments.docker import DockerEnvironment
from minimal_agent.models import OpenAIModel


def checked_output(env: DockerEnvironment, command: str) -> str:
    result = env.execute({"command": command})
    if result["returncode"]:
        raise RuntimeError(f"{command} failed: {result['exception_info']} {result['output']}")
    return result["output"]


def run_instance(instance: dict, model: OpenAIModel, agent_config: dict, output_root: Path) -> Path:
    instance_id = instance["instance_id"]
    base = instance["base_commit"]
    if not re.fullmatch(r"[\w.-]+__[\w.-]+-\d+", instance_id) or not re.fullmatch(r"[a-f0-9]{40}", base):
        raise ValueError("Invalid SWE-bench instance_id or base_commit")

    out = output_root / instance_id
    out.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=out))
    print(f"Run directory: {run_dir}", flush=True)

    image_id = instance_id.lower().replace("__", "_1776_")
    env = DockerEnvironment(
        image=f"swebench/sweb.eval.x86_64.{image_id}:latest", cwd="/testbed",
        run_args=["--rm", "--platform", "linux/amd64"], pull_timeout=1200, timeout=120,
        interpreter=["bash", "-lc", 'source /opt/miniconda3/bin/activate testbed && exec bash -c "$1"', "agent"],
    )
    try:
        checked_output(env, f"git merge-base --is-ancestor {base} HEAD")
        start_commit = checked_output(env, "git rev-parse HEAD").strip()
        if checked_output(env, "git status --porcelain"):
            raise RuntimeError("The instance image must start with a clean working tree")
        info = checked_output(env, "uname -s && uname -r && uname -v && uname -m").splitlines()
        system = dict(zip(("system", "release", "version", "machine"), info, strict=True))
        agent = DefaultAgent(model, env, **(agent_config | {"output_path": run_dir / "trajectory.json"}))
        result = agent.run(instance["problem_statement"], **system)
        if result["exit_status"] != "Submitted":
            raise RuntimeError(f"Agent exited with {result['exit_status']}")

        # Include new files and agent-created commits, relative to the image's initial checkout.
        checked_output(env, "git add -A")
        patch = checked_output(env, f"git diff --cached --no-ext-diff --no-color --binary {start_commit} --")
        if not patch:
            raise RuntimeError("Agent submitted an empty patch")
        prediction = {"instance_id": instance_id, "model_name_or_path": model.config.model, "model_patch": patch}
        path = run_dir / "predictions.jsonl"
        path.write_text(json.dumps(prediction, ensure_ascii=False) + "\n", encoding="utf-8")
        return path
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path, help="JSON object with instance_id, base_commit and problem_statement")
    args = parser.parse_args()
    instance = json.loads(args.instance.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")
    config = yaml.safe_load((root / "minimal_agent/config/default.yaml").read_text(encoding="utf-8"))
    model = OpenAIModel(**config["model"])
    path = run_instance(instance, model, config["agent"], root / "outputs/swebench")
    print(f"Prediction: {path}")


if __name__ == "__main__":
    main()
