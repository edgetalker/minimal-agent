import logging
import os
import uuid
import subprocess
from typing import Any
from pydantic import BaseModel

from minimal_agent.exceptions import Submitted, TimeExceeded

class DockerEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    env: dict[str, str] = {}
    forward_env: list[str] = []
    timeout: int = 30
    executable: str = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    run_args: list[str] = ["--rm"]
    container_timeout: str = "2h"
    pull_timeout: int = 120
    interpreter: list[str] = ["bash", "-lc"]

class DockerEnvironment:
    def __init__(
        self,
        *,
        config_class: type = DockerEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        """This class execute bash command in a Docker container using direct docker commands"""
        self.logger = logger or logging.getLogger("minimalagent.environment")
        self.container_id: str | None = None
        self.config = config_class(**kwargs)
        self._start_container()

    def _start_container(self):
        """Start the container and store its ID for subsequent commands."""
        container_name = f"minimalagent-{uuid.uuid4().hex}"
        cmd = [
            self.config.executable,
            "run",
            "-d",
            "--name",
            container_name,
            "-w",
            self.config.cwd,
            *self.config.run_args,
            self.config.image,
            "sleep",
            self.config.container_timeout,
        ]
        self.logger.debug("Starting container %s from %s", container_name, self.config.image)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.config.pull_timeout, check=True,
            )
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            self.container_id = container_name
            self.close()
            raise
        self.logger.info(f"Started container {container_name} with ID {result.stdout.strip()}")
        self.container_id = result.stdout.strip()

    def close(self) -> None:
        if self.container_id is None:
            return
        result = subprocess.run(
            [self.config.executable, "rm", "-f", self.container_id],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode and "no such container" not in result.stderr.lower():
            raise RuntimeError(f"Could not remove container {self.container_id}: {result.stderr.strip()}")
        self.container_id = None

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the Docker container and return the result as a dict"""
        command = action.get("command", "")
        cwd = cwd or self.config.cwd
        if not self.container_id:
            raise RuntimeError("Docker environment is closed")

        cmd = [self.config.executable, "exec", "-w", cwd]

        # 补充环境变量
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self.container_id, *self.config.interpreter, command])

        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=self.config.timeout if timeout is None else timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except subprocess.TimeoutExpired as exc:
            partial = exc.output or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            detail = f"Docker command timed out after {exc.timeout} seconds"
        except OSError as e:
            return {
                "output": "",
                "returncode": -1,
                "exception_info": e.strerror or type(e).__name__,
                "extra": {"exception_type": type(e).__name__},
            }
        else:
            output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
            self._check_finished(output)
            return output

        # Remove the container after leaving the handler so cleanup errors cannot chain the raw argv.
        self.close()
        raise TimeExceeded({
            "role": "exit",
            "content": detail,
            "extra": {
                "exit_status": "TimeExceeded", "submission": "", "returncode": -1,
                "output": partial, "exception_info": detail, "exception_type": "TimeoutExpired",
            },
        })

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

