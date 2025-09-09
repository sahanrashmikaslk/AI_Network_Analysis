"""
Local LLM Provider for testing without cloud credentials
This creates a mock LLM for development/testing purposes
"""

from typing import Any, Dict
import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List

class MockChatModel(BaseChatModel):
    """Mock chat model for testing without real LLM"""
    
    model: str = "mock-model"
    temperature: float = 0.7
    max_output_tokens: int = 1024
    
    @property
    def _llm_type(self) -> str:
        return "mock"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Any:
        # Mock response based on the input
        last_message = messages[-1].content if messages else ""
        
        # Generate a mock response for anomaly detection
        if "network metrics" in last_message.lower() or "anomaly" in last_message.lower():
            mock_response = '''
            {
                "findings": [
                    {
                        "metric_name": "cpu_percent",
                        "anomaly_type": "spike",
                        "severity": "medium",
                        "confidence": 0.85,
                        "description": "CPU usage spike detected above normal baseline",
                        "affected_timeframe": "last 15 minutes",
                        "baseline_value": 25.0,
                        "anomalous_value": 78.5,
                        "deviation_percentage": 214.0
                    }
                ],
                "recommendations": [
                    {
                        "category": "performance",
                        "priority": "medium",
                        "title": "Investigate CPU usage spike",
                        "description": "Review running processes and resource consumption patterns",
                        "expected_impact": "Improved system stability and performance",
                        "implementation_effort": "low"
                    }
                ],
                "risk_assessment": {
                    "overall_risk_level": "medium",
                    "risk_factors": ["Elevated CPU usage", "Potential resource contention"],
                    "potential_impacts": ["System slowdown", "Service degradation"],
                    "urgency_score": 0.6
                },
                "summary": "Detected moderate CPU usage anomaly. Recommend monitoring and investigation.",
                "analysis_confidence": 0.8
            }
            '''
        else:
            mock_response = "This is a mock response from the local LLM provider for testing purposes."
        
        from langchain_core.outputs import ChatGeneration, ChatResult
        
        message = AIMessage(content=mock_response)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

def build_chat_model(model_config: Dict[str, Any] = None) -> MockChatModel:
    """
    Build a mock chat model for local testing
    """
    return MockChatModel()
