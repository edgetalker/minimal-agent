import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from dotenv import load_dotenv

from minimal_agent.agent import DefaultAgent
from minimal_agent.models import OpenAIModel
from minimal_agent.environments.local import LocalEnvironment


repo = Path(__file__).resolve().parent
load_dotenv(repo / ".env")

config_path = repo / "minimal_agent" / "config" / "default.yaml"
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if not isinstance(config, dict) or not isinstance(config.get("agent"), dict):
    raise ValueError(f"配置文件必须包含 agent 配置字典：{config_path}")
agent_config = dict(config["agent"])

# YAML 中的相对输出路径按仓库目录解析，未指定时沿用默认路径。
trajectory = Path(agent_config.get("output_path") or "outputs/trajectory.json")
if not trajectory.is_absolute():
    trajectory = repo / trajectory
agent_config["output_path"] = trajectory

# 每次使用新的目录，避免上次修复影响本次测试。
workdir = Path(tempfile.mkdtemp(prefix="minimal-agent-live-"))

(workdir / "calculator.py").write_text(
    "def add(a, b):\n"
    "    return a - b\n",
    encoding="utf-8",
)

test_source = """\
import unittest
from calculator import add

class TestCalculator(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(add(2, 3), 5)

    def test_negative(self):
        self.assertEqual(add(-2, -3), -5)

    def test_zero(self):
        self.assertEqual(add(0, 7), 7)
"""
(workdir / "test_calculator.py").write_text(
    test_source, encoding="utf-8"
)

# 使用运行本脚本的 Python，避免 shell 中的 Python 环境不同。
test_command = f"{shlex.quote(sys.executable)} -m unittest -v"

model=OpenAIModel(
        **config["model"],
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL"),
)

agent = DefaultAgent(
    model=model,
    env=LocalEnvironment(),
    **agent_config,
)

task = f"""\
请按以下顺序完成任务，每个阶段使用独立的一轮工具调用：

1. 查看 calculator.py 和 test_calculator.py。
2. 执行 `{test_command}`，观察失败结果，此时不要修改代码。
3. 根据失败结果修复 calculator.py，不允许修改测试文件。
4. 再次执行 `{test_command}`，确认三个测试全部通过。
5. 测试通过后，单独执行完成标记命令。

本次测试每轮只调用一次 bash，等待工具结果后再进行下一步。
只允许修改当前目录中的 calculator.py。
不要提前提交；如果修复后测试仍失败，继续修复。
"""

print(f"配置文件：{config_path}", flush=True)
print(f"工作目录：{workdir}", flush=True)
print(f"轨迹文件：{trajectory}", flush=True)

# 当前 LocalEnvironment 继承进程工作目录。
previous_cwd = Path.cwd()
try:
    os.chdir(workdir)
    # 当前 Environment 尚未提供模板变量，由测试入口补齐系统信息。
    result = agent.run(task, **platform.uname()._asdict())
finally:
    os.chdir(previous_cwd)

# 验证任务结果，独立运行测试，不只相信模型的描述。
assert result["exit_status"] == "Submitted", result
assert (workdir / "test_calculator.py").read_text(
    encoding="utf-8"
) == test_source, "模型修改了测试文件"

check = subprocess.run(
    [sys.executable, "-m", "unittest", "-v"],
    cwd=workdir,
    capture_output=True,
    text=True,
)
assert check.returncode == 0, check.stdout + check.stderr

# 验证保存的 JSON、统计和内存中的消息一致。
data = json.loads(trajectory.read_text(encoding="utf-8"))
messages = data["messages"]

assert messages == agent.messages, "文件与内存中的消息不一致"
assert data["info"]["exit_status"] == "Submitted"
assert data["info"]["model_stats"]["api_calls"] == agent.n_calls
assert messages[0]["role"] == "system"
assert messages[1]["role"] == "user"
assert messages[-1]["role"] == "exit"

# 验证工具调用与观察结果一一对应，且调用先于结果。
calls = {}
observations = {}

for message in messages:
    if message.get("object") == "response":
        for action in message["extra"]["actions"]:
            call_id = action["tool_call_id"]
            assert call_id not in calls, f"重复 call_id：{call_id}"
            calls[call_id] = action["command"]

    elif message.get("type") == "function_call_output":
        call_id = message["call_id"]
        assert call_id in calls, f"找不到对应调用：{call_id}"
        assert call_id not in observations, f"重复工具结果：{call_id}"
        observations[call_id] = json.loads(message["output"])

# 你的提交命令直接抛出 Submitted，因此它没有普通工具结果。
finish_command = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
finish_ids = {
    call_id
    for call_id, command in calls.items()
    if command.strip() == finish_command
}
assert len(finish_ids) == 1, "应当有一次独立的提交调用"
assert set(calls) - set(observations) == finish_ids, "普通工具结果缺失"

# 确认轨迹记录了测试从失败到成功的过程。
test_results = [
    output
    for call_id, output in observations.items()
    if "unittest" in calls[call_id]
]
assert len(test_results) >= 2, "没有记录修复前后的两次测试"
assert test_results[0]["returncode"] != 0, "没有先运行失败测试"
assert test_results[-1]["returncode"] == 0, "最后一次测试没有通过"

print(f"验证通过，模型调用次数：{agent.n_calls}")
print(f"轨迹保留在：{trajectory}")
