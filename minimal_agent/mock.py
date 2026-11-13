from minimal_agent.exceptions import InterruptAgentFlow

class MockModel:
    def __init__(self):
        self.calls = 0

    def format_observation_messages(self, message: dict, outputs: list, template_vars: dict) -> list[dict]:
        """组装工具调用信息"""
        return [
            {
                "role": "user",
                "content": str(output),
            }
            for output in outputs
        ]

    def format_message(self, role: str, content: str) -> dict:
        """组装单词上下文信息"""
        return {
            "role": role,
            "content": content
        }

    def query(self, messages: dict[str]) -> dict:
        """发送 query"""
        self.calls += 1

        if self.calls == 1:
            return {
                "role": "assistant",
                "content": "Execute test action",
                "extra": {
                    "cost": 0.01,
                    "actions": ["echo hello"],
                },
            }

        # 第二次调用：模拟 Agent 完成任务
        raise InterruptAgentFlow(
            {
                "role": "exit",
                "content": "Submitted",
                "extra": {
                    "exit_status": "Submitted",
                    "submission": "Task completed",
                },
            }
        )

class MockEnvironment:
    """负责工具执行逻辑"""
    def execute(self, action: str) -> dict:
        return {
            "returncode": 0,
            "output": f"Executed: {action}",
        }