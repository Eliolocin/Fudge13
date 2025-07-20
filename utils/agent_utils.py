"""Agent utilities for agentic LLM functionality with Google Search grounding and Thought Summaries"""

from google import genai
from google.genai import types
from typing import List, Dict, Any

def create_google_search_tool() -> types.Tool:
    """
    Create Google Search grounding tool for enhanced translation evaluation.
    
    This tool allows the LLM to search for real-time information about:
    - Translation guidelines and best practices
    - Cultural context and references
    - Language usage patterns and examples
    - Current Filipino language standards
    
    Returns:
        types.Tool: Configured Google Search grounding tool
    """
    return types.Tool(
        google_search=types.GoogleSearch()
    )

def create_thinking_config(include_thoughts: bool = True) -> types.ThinkingConfig:
    """
    Create thinking configuration for capturing thought summaries.
    
    Enables the LLM to provide insight into its reasoning process when
    evaluating translation quality across the 6 criteria.
    
    Args:
        include_thoughts (bool): Whether to include thought summaries in response
        
    Returns:
        types.ThinkingConfig: Configuration for thought summary capture
    """
    return types.ThinkingConfig(
        include_thoughts=include_thoughts
    )

def create_agentic_config(
    include_google_search: bool = True,
    include_thoughts: bool = True,
    response_mime_type: str = "application/json",
    response_schema: Any = None
) -> types.GenerateContentConfig:
    """
    Create comprehensive configuration for agentic LLM behavior.
    
    Combines Google Search grounding, thought summaries, and structured output
    for enhanced translation evaluation capabilities.
    
    Args:
        include_google_search (bool): Enable Google Search grounding
        include_thoughts (bool): Enable thought summary capture
        response_mime_type (str): MIME type for structured output
        response_schema (Any): Pydantic schema for response validation
        
    Returns:
        types.GenerateContentConfig: Complete agentic configuration
    """
    tools = []
    
    # Add Google Search grounding if enabled
    if include_google_search:
        tools.append(create_google_search_tool())
    
    config_params = {}
    
    # Add tools if any are configured
    if tools:
        config_params['tools'] = tools
    
    # Add thinking configuration if enabled
    if include_thoughts:
        config_params['thinking_config'] = create_thinking_config(include_thoughts)
    
    # Add structured output configuration
    if response_mime_type and response_schema:
        config_params['response_mime_type'] = response_mime_type
        config_params['response_schema'] = response_schema
    
    return types.GenerateContentConfig(**config_params)

def extract_thought_summary(response) -> Dict[str, Any]:
    """
    Extract thought summary and main content from agentic response.
    
    Separates the LLM's reasoning process from the final structured judgment
    to provide transparency in the evaluation process.
    
    Args:
        response: Response object from Gemini API
        
    Returns:
        Dict containing:
        - thought_summary (str): The LLM's reasoning process
        - main_content (str): The structured judgment response
        - has_thoughts (bool): Whether thought summary was found
    """
    thought_summary = ""
    main_content = ""
    has_thoughts = False
    
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
            for part in candidate.content.parts:
                if hasattr(part, 'thought') and part.thought:
                    # This part contains thought content
                    thought_summary += part.text + "\n"
                    has_thoughts = True
                else:
                    # This part contains the main response
                    main_content += part.text
    
    # Fallback: if no structured parts, use the main response text
    if not main_content and hasattr(response, 'text'):
        main_content = response.text
    
    return {
        "thought_summary": thought_summary.strip(),
        "main_content": main_content.strip(),
        "has_thoughts": has_thoughts
    }

# Supported agentic models - only Gemini 2.5 series support full agentic features
AGENTIC_SUPPORTED_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-05-20"
]

def is_agentic_model_supported(model_name: str) -> bool:
    """
    Check if a model supports agentic features (Google Search + Thought Summaries).
    
    Args:
        model_name (str): Name of the model to check
        
    Returns:
        bool: True if model supports agentic features
    """
    return model_name in AGENTIC_SUPPORTED_MODELS