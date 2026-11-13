from openai import OpenAI
import json

class OpenAIModel:
    def __init__(
        self, 
        model: str, 
        base_url: str | None = None, 
        api_key: str | None = None,
    ):
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=30.0
        )

        self.tools = [
            {
                "type": "function",
                "name": "bash",
                "description": "执行 bash 命令，返回命令输出和退出码。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "需要执行的 bash 命令",
                        }
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]
    def _parse_actions(self, response: dict) -> list[dict]:
        """从 Responses API 的结构化输出中提取工具"""
        actions = []

        for item in response.output:
            if item.type != "function_call":
                continue

            if item.name != "bash":
                raise ValueError(f"未知工具：{item.name}")

            # 解析调用参数
            arguments = json.loads(item.arguments)
            command = arguments.get("command")
            if not isinstance(command, str) or not command.strip():
                raise ValueError("command 必须是非空字符串")

            actions.append({
                "name": item.name,
                "command": command,
                "call_id": item.call_id,
            })
            
        return actions

    
    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        response = self.client.responses.create(
            model=self.model,
            input=messages,
            tools=self.tools,  
        )

        if response.status != "completed":
            raise RuntimeError(
                f"模型响应未完成：status={response.status}, "
                f"details={response.incomplete_details}, "
                f"error={response.error}"
            )

        return {
            # 普通回复；模型调用工具时可能为空
            "content": response.output_text,

            # 给 agent 执行的动作列表
            "actions": self._parse_actions(response),

            # 完整输出，用于维护后续对话
            "output": [
                item.model_dump(exclude_none=True)
                for item in response.output
            ],
        }

    def format_message(self, **kwargs) -> dict:
        return dict(kwargs)

    def format_observation_messages(
            self, message: dict, outputs: list[dict], template_vars: dict | None
        ) -> list[dict]:
        """组装 function call output """
        actions = message["actions"]

        return [
            {
                "type": "function_call_output",
                "call_id": action["call_id"],
                "output": json.dumps(output, ensure_ascii=False)
            }
            for action, output in zip(actions, outputs)
        ]
        