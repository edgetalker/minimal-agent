from typing import Any
import subprocess
import os

from minimal_agent.exceptions import Submitted

class LocalEnvironment:
    def execute(self, action: dict) -> dict[str, Any]:
        try:
            result = subprocess.run(
                action["command"],
                shell=True,
                text=True,
                env=os.environ,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
            output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
            # 检查是否为退出结果
            self._check_finished(output)
            return output
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")

            return {
                "output": output,
                "returncode": -1,
                "exception_info": "命令执行超过 30 秒",
            }
        
    def _check_finished(self, output: dict):
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