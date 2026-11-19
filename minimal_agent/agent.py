import json
import logging
import time
import traceback
from pathlib import Path

from jinja2 import Template, StrictUndefined
from pydantic import BaseModel

from minimal_agent.exceptions import InterruptAgentFlow, FormatError, LimitsExceeded, TimeExceeded
from minimal_agent import Model, Environment


def _compact_response_usage(response: dict) -> dict:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return response

    compact_usage = {
        key: usage[key]
        for key in ("input_tokens", "output_tokens", "total_tokens")
        if key in usage
    }
    for details_key, token_key in (
        ("input_tokens_details", "cached_tokens"),
        ("output_tokens_details", "reasoning_tokens"),
    ):
        details = usage.get(details_key)
        if isinstance(details, dict) and token_key in details:
            compact_usage[details_key] = {token_key: details[token_key]}

    return {**response, "usage": compact_usage}


def _serialize_message(message: dict) -> dict:
    if message.get("object") == "response":
        return _compact_response_usage(message)

    extra = message.get("extra")
    if isinstance(extra, dict) and extra.get("interrupt_type") == "FormatError":
        response = extra.get("response")
        if isinstance(response, dict):
            return {**message, "extra": {**extra, "response": _compact_response_usage(response)}}
    return message


class AgentConfig(BaseModel):
    # 配置类使用 pydantic 
    system_template: str
    instance_template: str
    step_limit: int = 0
    cost_limit: float = 3.0
    wall_time_limit_seconds: int = 0
    max_consecutive_format_errors: int = 3
    output_path: Path | None = None
    
class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        """
        初始化以下组件：
        1）消息列表
        2）工具执行环境
        3）模型
        """
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        # 保存Agent运行时的动态变量，并将其注入 Jinja2 Prompt 模板
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0
        self.n_consecutive_format_errors = 0
        self._start_time = time.time()

    def get_template_vars(self, **kwargs) -> dict:
        return {
            **self.extra_template_vars,
            **kwargs,
        }

    def _render_template(self, template: str) -> str:
        return Template(
            template,
            undefined=StrictUndefined,
        ).render(**self.get_template_vars())
    
    def add_messages(self, *messages: dict) -> list[dict]:
        """组装消息列表"""
        self.logger.debug(messages) 
        self.messages.extend(messages)
        return list(messages) 

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role= "exit",
                content= str(e),
                extra= {
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "trace_back": traceback.format_exc(),
                },
            )
        )

    def run(self, task: str = "", **kwargs) -> dict:
        """运行agent直到终止，返回字典"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []

        # 装配 sp 和 up
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )

        # Agent Loop
        while True:
            try:
                self.step()
                self.n_consecutive_format_errors = 0 # 每一步重置
            except FormatError as e:
                self.cost += e.messages[0].get("extra", {}).get("cost", 0.0)
                self.n_consecutive_format_errors += 1
                if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                    self.add_messages(
                        *e.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                        },
                    )
                else:
                    self.add_messages(*e.messages)
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                # 保存agent执行轨迹
                self.save(self.config.output_path)
            # 退出条件
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    def step(self) -> list[dict]:
        """向 LM 发送消息，执行一步"""
        return self.execute_actions(self.query())
    
    def query(self) -> dict:
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time):
            raise TimeExceeded(
                {
                    "role": "exit",
                    "content": "TimeExceeded",
                    "extra": {"exit_status": "TimeExceeded", "submission": ""},
                }
            )
        self.n_calls += 1
        message = self.model.query(self.messages)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        """从agent返回信息中提取工具参数并执行，返回 function_call_output """
        outputs = [self.env.execute(action) for action in message.get("extra", {}).get("actions", [])]
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def serialize(self) -> dict:
        """将 agent state 序列化为 json 字典"""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})

        # 组织完整的状态字段
        agent_data = {
            "trajectory_format": "minimal-agent-v1",
            "messages": [_serialize_message(message) for message in self.messages],
            "info": {
                "model_stats": {
                    "api_calls": self.n_calls,
                    "instance_cost": self.cost,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json")
                },
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            }
        }
        # TODO 当前只保存 agent 执行轨迹，没有 model、env 信息
        return agent_data

    def save(self, path: Path | None) -> dict:
        data = self.serialize()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        return data
