import json
import time

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from minimal_agent.exceptions import FormatError

from jinja2 import StrictUndefined, Template

class TokenPricing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: float = Field(ge=0, allow_inf_nan=False)
    cached_input: float = Field(ge=0, allow_inf_nan=False)
    output: float = Field(ge=0, allow_inf_nan=False)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    pricing: TokenPricing | None = None
    observation_template: str

class OpenAIModel:
    def __init__(
        self, 
        *,
        base_url: str | None = None, 
        api_key: str | None = None,
        **kwargs,
    ):
        self.config = ModelConfig(**kwargs)
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
        # 需要引入 Template
        raise FormatError({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text","text":
                f"工具调用格式错误：{error}\n"
                "本轮未执行任何命令，请重新调用 bash 工具。\n"
                '参数格式：{"command": "需要执行的命令"}\n'
                "如果任务已经完成，请单独执行：\n"
                "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            }],
            "extra": {"interrupt_type": "FormatError"}
        })

    def _parse_actions(self, response: dict) -> list[dict]:
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
                "command": command,
                "tool_call_id": item.call_id,
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

        # 2. 获取模型响应并计算费用
        response = self.client.responses.create(
            model=self.config.model,
            input=input,
            tools=self.tools,  
        )
        
        cost_output = self._calculate_cost(response.usage)

        # 3. 解析回答中的工具调用
        try:
            actions = self._parse_actions(response)
        except FormatError as e:
            e.messages[0]["extra"].update(cost_output)
            try:
                e.messages[0]["extra"]["response"] = response.model_dump(mode="json")
            except Exception:
                # model_dump failed (e.g. unserializable object); fall back to repr
                # so the spec contract ("response MUST be persisted") holds unconditionally.
                e.messages[0]["extra"]["response"] = repr(response)
            raise

        # 4. 组装为模型回答
        message = response.model_dump(mode="json", by_alias=True, exclude_none=True) if hasattr(response, "model_dump") else dict(response)
        message["extra"] = {
            "actions": actions,
            **cost_output,
            "timestamp": time.time()
        }
        return message

    def _calculate_cost(self, usage: dict) -> dict[str, float]:
        pricing = self.config.pricing
        cached_tokens = usage.input_tokens_details.cached_tokens
        uncached_tokens = usage.input_tokens - cached_tokens

        cost = (uncached_tokens * pricing.input
                + cached_tokens * pricing.cached_input
                + usage.output_tokens * pricing.output
            ) / 1000000
        return {"cost": cost}

    def format_message(self, **kwargs) -> dict:
        return dict(kwargs)

    def format_observation_messages(
            self, message: dict, outputs: list[dict], template_vars: dict | None = None
        ) -> list[dict]:
        """格式化工具调用执行结果"""
        actions = message.get("extra", {}).get("actions", [])

        # 1. 补齐没有执行结果的工具调用
        not_executed = {"output": "", "returncode": -1, "exception_info": "action was not executed"}
        padded_outputs = outputs + [not_executed] * (len(actions)-len(outputs))

        # 2. 模版渲染模型看到的内容
        results = []
        for action, output in zip(actions, padded_outputs):
            content = Template(self.config.observation_template, undefined=StrictUndefined).render(
                output=output, **(template_vars or {})
            )
            msg: dict = {
                "extra": {
                    "raw_output": output.get("output", ""),
                    "returncode": output.get("returncode"),
                    "timestamp": time.time(),
                    "exception_info": output.get("exception_info"),
                    **output.get("extra", {}),
                },
            }
            if "tool_call_id" in action:
                msg["type"] = "function_call_output"
                msg["call_id"] = action["tool_call_id"]
                msg["output"] = content
            else:
                msg["type"] = "message"
                msg["role"] = "user"
                msg["content"] = [{"type": "input_text", "text": content}]
            results.append(msg)
        return results
        