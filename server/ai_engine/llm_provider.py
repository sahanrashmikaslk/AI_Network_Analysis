"""
LLM Provider utility (env-only)
Uses a fine-tuned Vertex AI model deployed to an endpoint, or falls back to local mock for testing.

Required env vars for Vertex AI:
- VERTEX_ENDPOINT_ID: your deployed model endpoint ID (the numeric ID from your deployment)

Optional env vars:
- GOOGLE_CLOUD_PROJECT: GCP project id (ADC can also provide)
- VERTEXAI_LOCATION: Vertex AI region (default: us-central1)
- GOOGLE_APPLICATION_CREDENTIALS: path to service account json for local dev
"""

from typing import Any, Dict
import os
import logging

logger = logging.getLogger(__name__)

def build_chat_model(model_config: Dict[str, Any] = None):
    """
    Build a chat model using environment variables, with fallback to local mock.
    Uses a fine-tuned model deployed to a Vertex AI endpoint if credentials available.
    Args:
        model_config: Ignored - all config comes from env vars
    Returns:
        ChatVertexAI instance for the deployed endpoint, or MockChatModel for testing
    """
    endpoint_id = os.getenv('VERTEX_ENDPOINT_ID')
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    location = os.getenv('VERTEXAI_LOCATION', 'us-central1')
    
    # Try to use Vertex AI if credentials are available
    if endpoint_id and project:
        try:
            from langchain_google_vertexai import ChatVertexAI
            
            # Use the full endpoint path as the model name
            full_endpoint = f"projects/{project}/locations/{location}/endpoints/{endpoint_id}"
            
            # Create ChatVertexAI with minimal parameters to avoid validation errors
            return ChatVertexAI(
                model=full_endpoint,
                project=project,
                location=location,
                temperature=0.7,
                max_output_tokens=1024
            )
        except Exception as e:
            logger.warning(f"Could not initialize Vertex AI model: {e}")
            logger.info("Falling back to local mock model for development")
    else:
        logger.info("No Vertex AI credentials found, using local mock model")
    
    # Fallback to local mock model
    from .llm_provider_local import build_chat_model as build_mock_model
    return build_mock_model(model_config)
