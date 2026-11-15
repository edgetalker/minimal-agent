import json
import time

from openai import OpenAI
from minimal_agent.exceptions import FormatError

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
                },
            }
        ]

    def _raise_format_error(self, error: str):
        raise FormatError({
            "role": "user",
            "content": (
                f"工具调用格式错误：{error}\n"
                "本轮未执行任何命令，请重新调用 bash 工具。\n"
                '参数格式：{"command": "需要执行的命令"}\n'
                "如果任务已经完成，请单独执行：\n"
                "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            ),
        })


    def _parse_actions(self, response: dict) -> list[dict]:
        # TODO: 对比 models/utils/actions_toolcall_response.py
        """从 Responses API 的结构化输出中提取工具"""
        calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]

        if not calls:
            self._raise_format_error("没有找到工具调用。")

        actions = []

        for item in calls:
            if item.name != "bash":
                self._raise_format_error(
                    f"未知工具{item.name!r}, 当前只支持 bash。"
                )

            try:
                arguments = json.loads(item.arguments)
            except (json.JSONDecodeError, TypeError):
                self._raise_format_error(
                    "arguments 不是合法的 JSON 参数。"
                )

            if not isinstance(arguments, dict):
                self._raise_format_error(
                    "arguments 必须是 JSON 对象。"
                )

            command = arguments.get("command")
            if not isinstance(command, str) or not command.strip():
                self._raise_format_error(
                    "缺少有效的 command，必须提供非空字符串。"
                )
            
            actions.append({
                "name": item.name,
                "command": command,
                "call_id": item.call_id,
            })

        return actions

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        """生成新的API输入列表"""
        result = []
        for msg in messages:
            if msg.get("object") == "response":
                for item in msg.get("output", []):
                    result.append({k: v for k, v in item.items() if k != "extra"})
            else:
                result.append({k: v for k, v in msg.items() if k != "extra"})
        return result
    
    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        # 1. 将消息字段转换为 API 的输入字段
        input = self._prepare_messages_for_api(messages)

        # 2. 获取模型响应（重试+计费字段计算）
        response = self.client.responses.create(
            model=self.model,
            input=input,
            tools=self.tools,  
        )

        if response.status != "completed":
            raise RuntimeError(
                f"模型响应未完成：status={response.status}, "
                f"details={response.incomplete_details}, "
                f"error={response.error}"
            )
        # 3. 解析回答中的工具调用
        try:
            actions = self._parse_actions(response)
        except FormatError as exc:
            metadata = exc.messages[0].setdefault("extra", {})
            try:
                metadata["response"] = response.model_dump(mode="json")
            except Exception:
                metadata["response"] = repr(response)
            raise

        # 4. 组装为模型回答
        message = response.model_dump(mode="json", by_alias=True, exclude_none=True) if hasattr(response, "model_dump") else dict(response)
        message["extra"] = {
            "actions": actions,
            # **cost_output,
            "timestamp": time.time()
        }
        return message
    
    def format_message(self, **kwargs) -> dict:
        return dict(kwargs)

    def format_observation_messages(
            self, message: dict, outputs: list[dict], template_vars: dict | None
        ) -> list[dict]:
        """格式化工具调用执行结果"""
        actions = message.get("extra", {}).get("actions", [])
        # TODO 需要与model解耦 并应用observation 模版
        return [
            {
                "type": "function_call_output",
                "call_id": action["call_id"],
                "output": json.dumps(output, ensure_ascii=False)
            }
            for action, output in zip(actions, outputs)
        ]
        