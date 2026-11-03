"""
Model Configuration Schema
This module defines the Pydantic model for configuring LLM and VLM adapters.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union


class ModelConfig(BaseModel):
    """
    Configuration for Model Adapters (LLM and VLM).
    
    This schema matches the full payload from model_config.json and includes
    all fields needed for model management, configuration, and execution.
    """
    
    # Core identification fields
    model_id: str = Field(..., description="Unique identifier for the model instance")
    model_type: Optional[str] = Field(None, description="Type of the model (third_party, self_hosted)")
    model_name: str = Field(..., description="Name of the model (e.g., 'gpt-4o', 'gemini-pro-vision')")
    model_description: Optional[str] = Field(None, description="Optional description of the model")
    model_provider: str = Field(..., description="Provider name (e.g., 'OpenAI', 'Google', 'Gemini')")
    model_version: Optional[str] = Field(None, description="Version of the model")
    
    # Fallback and moderation
    fallback_model: Optional[str] = Field(None, description="Fallback model to use if primary fails")
    pre_moderations: List[str] = Field(default_factory=list, description="List of pre-moderation checks")
    post_moderations: List[str] = Field(default_factory=list, description="List of post-moderation checks")
    
    # Authentication and endpoints
    api_key: Optional[str] = Field(None, description="API key for authentication")
    api_endpoint: Optional[str] = Field(None, description="API endpoint (deprecated, use endpoint_url)")
    endpoint_url: Optional[str] = Field(None, description="Full endpoint URL (for self-hosted or custom endpoints)")
    authentication_type: Optional[str] = Field(None, description="Authentication type (bearer, api_key, none)")

    # Vertex AI (Gemini on GCP project billing). Alternative to api_key: a GCP
    # service-account JSON exchanged for an OAuth2 token. project id defaults to
    # the one in the JSON; location is the Vertex region.
    service_account_json: Optional[str] = Field(None, description="GCP service-account JSON (Vertex AI auth)")
    vertex_location: Optional[str] = Field(None, description="Vertex AI region, e.g. 'us-central1' (default) or 'global'")
    vertex_project: Optional[str] = Field(None, description="GCP project id override (else taken from the service-account JSON)")
    
    # Request configuration
    timeout: int = Field(180, description="Request timeout in seconds", ge=1, le=1800)
    max_retries: Optional[int] = Field(3, description="Maximum number of retries")
    rate_limit: Optional[int] = Field(None, description="Requests per minute limit")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens in response")
    
    # File and format limits
    max_image_size: Optional[int] = Field(None, description="Maximum image size in bytes")
    max_file_size: Optional[int] = Field(None, description="Maximum file size in bytes")
    supported_formats: Optional[List[str]] = Field(None, description="Supported image formats (jpg, png, etc.)")
    
    # Vision and multimodal capabilities
    vision_capabilities: List[str] = Field(default_factory=list, description="Vision capabilities (e.g., ['vision'])")
    # Vision image fidelity sent to the model. "auto" preserves prior behavior;
    # "low" caps tokens/cost for high-volume passes; "high" forces full tiling.
    image_detail: Optional[str] = Field("auto", description="Image detail hint: 'auto' | 'low' | 'high'")

    # Reasoning controls (o-series / native reasoning models). When set on a
    # reasoning-capable model, forwarded as the Responses API `reasoning.effort`.
    reasoning_effort: Optional[str] = Field(None, description="Reasoning effort: 'low' | 'medium' | 'high'")
    
    # Request/response formatting (for self-hosted models)
    request_format: Optional[str] = Field(None, description="Request payload format")
    response_format: Optional[str] = Field(None, description="Expected response format")
    request_schema: Optional[List[Dict[str, Any]]] = Field(None, description="Request schema definition")
    # Some model configurations (legacy / DB defaults) store response_schema as
    # a list of parameter definitions (array of {name,type}). Other parts of the
    # system expect a JSON-schema-like dict. Accept either form here to avoid
    # validation errors when converting DB model documents into ModelConfig.
    response_schema: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = Field(
        None, description="Response schema definition (list or dict)"
    )
    # When true, adapters should attempt to produce structured output
    # that conforms to `response_schema` (if provided). Adapters that
    # support provider-specific structured-output APIs can use those
    # mechanisms; otherwise they may fall back to an instruction-based
    # prompt to coerce JSON output.
    is_response_dynamic: Optional[bool] = Field(False, description="Whether to request structured output (JSON schema) from the model")
    
    # Additional configuration
    system_prompt: Optional[str] = Field(None, description="Optional system prompt for the model")
    document_url: Optional[str] = Field(None, description="Optional documentation URL for the model")
    
    model_config = {
        'protected_namespaces': ()
    }


def model_response_to_config(model_response: dict) -> ModelConfig:
    """
    Convert a ModelResponse dict to ModelConfig.
    
    Args:
        model_response: Dict from ModelResponse.model_dump()
        
    Returns:
        ModelConfig instance
        
    Example:
        >>> model_data = await collection.find_one({"model_id": "gpt-4o-main"})
        >>> config = model_response_to_config(model_data)
        >>> adapter = AdapterFactory.get_adapter(config)
    """
    # Extract only the fields that ModelConfig needs
    # This handles the conversion from DB schema to adapter config
    config_data = {
        'model_id': model_response.get('model_id'),
        'model_type': model_response.get('model_type'),
        'model_name': model_response.get('model_name'),
        'model_description': model_response.get('model_description'),
        'model_provider': model_response.get('model_provider'),
        'model_version': model_response.get('model_version'),
        'fallback_model': model_response.get('fallback_model'),
        'pre_moderations': model_response.get('pre_moderations', []),
        'post_moderations': model_response.get('post_moderations', []),
        'api_key': model_response.get('api_key'),
        'api_endpoint': model_response.get('api_endpoint'),
        'endpoint_url': model_response.get('endpoint_url'),
        'authentication_type': model_response.get('authentication_type'),
        'service_account_json': model_response.get('service_account_json'),
        'vertex_location': model_response.get('vertex_location'),
        'vertex_project': model_response.get('vertex_project'),
        'timeout': max(int(model_response.get('timeout') or 180), 180),
        'max_retries': model_response.get('max_retries', 3),
        'rate_limit': model_response.get('rate_limit'),
        'max_tokens': model_response.get('max_tokens'),
        'max_image_size': model_response.get('max_image_size'),
        'max_file_size': model_response.get('max_file_size'),
        'supported_formats': model_response.get('supported_formats'),
        'vision_capabilities': model_response.get('vision_capabilities', []),
        'image_detail': model_response.get('image_detail'),
        'reasoning_effort': model_response.get('reasoning_effort'),
        'request_format': model_response.get('request_format'),
        'response_format': model_response.get('response_format'),
        'request_schema': model_response.get('request_schema'),
        'response_schema': model_response.get('response_schema'),
        'is_response_dynamic': model_response.get('is_response_dynamic', False),
        'system_prompt': model_response.get('system_prompt'),
        'document_url': model_response.get('document_url'),
    }
    
    # Remove None values to allow defaults to take effect
    config_data = {k: v for k, v in config_data.items() if v is not None}
    
    return ModelConfig(**config_data)
