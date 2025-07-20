"""Agentic LLM client implementation with Google Search grounding and Thought Summaries"""

import os
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel
from google import genai
from utils.agent_utils import (
    create_agentic_config, 
    extract_thought_summary, 
    is_agentic_model_supported,
    AGENTIC_SUPPORTED_MODELS
)
from llms.prompt_engineered_judge_main import LLMClient, TranslationJudgment


class AgenticGeminiClient(LLMClient):
    """
    Agentic Google Gemini API client with enhanced capabilities:
    - Google Search grounding for real-time information
    - Thought Summary capture for transparency
    - Structured output for translation judgment
    
    Only supports Gemini 2.5 models (gemini-2.5-pro, gemini-2.5-flash)
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-pro"):
        super().__init__(api_key)
        
        # Validate model support for agentic features
        if not is_agentic_model_supported(model_name):
            raise ValueError(
                f"Model '{model_name}' does not support agentic features. "
                f"Supported models: {', '.join(AGENTIC_SUPPORTED_MODELS)}"
            )
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        
        print(f"[DEBUG] AgenticGeminiClient initialized with model: {model_name}")
    
    def generate_judgment(self, prompt: str) -> Dict[str, Any]:
        """
        Generate structured translation judgment using agentic Gemini with:
        - Google Search grounding for enhanced context
        - Thought summary capture for reasoning transparency
        - Structured output validation
        """
        print(f"[DEBUG] AgenticGeminiClient.generate_judgment called with prompt length: {len(prompt)}")
        print(f"[DEBUG] Prompt preview: {prompt[:100]}...")
        print(f"[DEBUG] EXACT PROMPT SENT TO AGENTIC GEMINI: {repr(prompt)}")
        
        try:
            print("[DEBUG] Calling Agentic Gemini API with Google Search grounding and thought summaries...")
            
            # Create agentic configuration with all features enabled
            config = create_agentic_config(
                include_google_search=True,
                include_thoughts=True,
                response_mime_type="application/json",
                response_schema=TranslationJudgment
            )
            
            # Generate content with agentic capabilities
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            
            print(f"[DEBUG] Agentic Gemini API response received")
            
            # Extract thought summary and main content
            extracted = extract_thought_summary(response)
            thought_summary = extracted["thought_summary"]
            main_content = extracted["main_content"]
            has_thoughts = extracted["has_thoughts"]
            
            print(f"[DEBUG] Thought summary extracted: {has_thoughts}")
            if has_thoughts:
                print(f"[DEBUG] Thought summary length: {len(thought_summary)}")
                print(f"[DEBUG] Thought summary preview: {thought_summary[:200]}...")
            
            print(f"[DEBUG] Main content length: {len(main_content)}")
            print(f"[DEBUG] Full main content: {main_content}")
            
            # Parse the structured response
            judgment_data = json.loads(main_content)
            print("[DEBUG] Successfully parsed JSON response")
            
            # Add thought summary to judgment data if available
            if has_thoughts and thought_summary:
                judgment_data["thought_summary"] = thought_summary
            
            # Validate with Pydantic (including thought_summary if present)
            judgment = TranslationJudgment(**judgment_data)
            print("[DEBUG] Successfully validated with Pydantic")
            
            return {
                "success": True,
                "data": judgment.model_dump(),
                "raw_response": main_content,
                "thought_summary": thought_summary if has_thoughts else None,
                "agentic_features": {
                    "google_search_enabled": True,
                    "thought_summary_captured": has_thoughts,
                    "model_name": self.model_name
                }
            }
            
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON decode error: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to parse JSON response: {str(e)}",
                "raw_response": main_content if 'main_content' in locals() else None,
                "thought_summary": thought_summary if 'thought_summary' in locals() else None
            }
        except Exception as e:
            print(f"[DEBUG] Agentic Gemini API error: {str(e)}")
            return {
                "success": False,
                "error": f"Agentic Gemini API error: {str(e)}"
            }


def create_agentic_llm_client(provider: str, model: str) -> Optional[LLMClient]:
    """
    Factory function to create agentic LLM client.
    Currently only supports Google Gemini 2.5 models with agentic features.
    
    Args:
        provider (str): LLM provider ("google" for agentic mode)
        model (str): Model name (must be Gemini 2.5 series)
        
    Returns:
        Optional[LLMClient]: AgenticGeminiClient if supported, None otherwise
        
    Raises:
        ValueError: If provider is not supported or model doesn't support agentic features
    """
    import os
    from dotenv import load_dotenv
    
    # Reload environment variables
    load_dotenv()
    
    print(f"[DEBUG] Creating agentic client for provider='{provider}', model='{model}'")
    
    if provider != "google":
        raise ValueError(f"Agentic mode currently only supports Google provider, got: {provider}")
    
    if not is_agentic_model_supported(model):
        raise ValueError(
            f"Model '{model}' does not support agentic features. "
            f"Supported models: {', '.join(AGENTIC_SUPPORTED_MODELS)}"
        )
    
    api_key = os.getenv("GOOGLE_API_KEY")
    print(f"[DEBUG] Google provider - API key exists: {bool(api_key)}")
    
    if not api_key or api_key.startswith("demo_key"):
        print("[DEBUG] No valid Google API key for agentic mode")
        raise ValueError("Agentic mode requires a valid Google API key")
    
    print("[DEBUG] Valid Google API key found, creating AgenticGeminiClient")
    try:
        return AgenticGeminiClient(api_key, model)
    except Exception as e:
        print(f"[DEBUG] Error creating AgenticGeminiClient: {e}")
        raise e