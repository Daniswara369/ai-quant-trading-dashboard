from abc import ABC, abstractmethod
from typing import Dict, Any, List
from core.schemas import AgentOutput, AgentContext

class BaseAgent(ABC):
    @abstractmethod
    def analyze(self, context: AgentContext) -> AgentOutput:
        """
        Analyze the given context and return a structured AgentOutput.
        """
        pass
