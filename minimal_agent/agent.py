import logging
import time
from pathlib import Path

from jinja2 import Template, StrictUndefined
from pydantic import BaseModel

from minimal_agent.exceptions import InterruptAgentFlow, FormatError, LimitsExceeded, TimeExceeded
from minimal_agent import Model, Environment

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
        '*'标记后续的参数只能按关键字传
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
        """将传入的多个message打包成一个tuple"""
        self.logger.debug(messages) 
        self.messages.extend(messages)
        return list(messages) 

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
            # 格式异常
            except FormatError as e:
                # 更新监控参数
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
            # 中断流
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.logger.exception("Unexpected agent error")
                # self.handle_uncaught_exception(e)
                raise
            #finally:
                # 保存agent执行轨迹
                # self.save(self.config.output_path)
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
        self.add_messages(*message["output"])
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        """执行消息列表中的动作，添加下一个状态的观察信息，并返回"""
        # 对应 model.query 返回结构
        outputs = [self.env.execute(action) for action in message.get("actions", [])]
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))