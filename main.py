from minimal_agent.agent import DefaultAgent
from minimal_agent.models import OpenAIModel
from minimal_agent.environments.local import LocalEnvironment

import os
import logging
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# 单独打开 Agent 的调试日志
logging.getLogger("agent").setLevel(logging.DEBUG)

def main():
    load_dotenv()
    agent = DefaultAgent(
        model = OpenAIModel(model=os.getenv("OPENAI_MODEL"), api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")),
        env = LocalEnvironment(),
        system_template = """You are a coding assistant.
    Use the bash tool to inspect files, modify code, and run tests.
    Every response must include at least one bash tool call.
    When the task is complete, call bash with exactly:
    echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
    Do not combine this command with other commands.""",
        instance_template = "Please solve: {{ task }}",
        step_limit = 15,
    )

    agent.run(task="""当前目录中有 calculator.py 和 test_calculator.py。
请完成以下任务：
1. 查看代码和测试。
2. 先运行 python -m unittest -v，确认失败原因。
3. 修复 calculator.py 中的错误，不要修改测试文件。
4. 再次运行测试，确认全部通过。
5. 最后说明修改内容和测试结果。

请使用 bash 工具实际操作。""")

if __name__ == "__main__":
    main()