from typing import Protocol, Any

class Model(Protocol):
    
    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None
    ) -> list[dict]: ...
    
    def format_message(self, **kwargs) -> dict: ...
            
    def query(self, messages: list[dict[str, str]], **kwargs) -> dict: ...            

class Environment(Protocol):
    def execute(self, action: dict) -> dict[str, Any]: ...


__all__ = [
    "Model",
    "Environment",
]