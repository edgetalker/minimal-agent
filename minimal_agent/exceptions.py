class InterruptAgentFlow(Exception):

    def __init__(self, *messages: dict):
        self.messages = messages
        super().__init__()

class Submitted(InterruptAgentFlow):
    """ Agent 主动结束任务"""
    
class FormatError(InterruptAgentFlow):
    """当 LM 输出不是指定的格式"""

class LimitsExceeded(InterruptAgentFlow):
    """"""

class TimeExceeded(LimitsExceeded):
    """"""