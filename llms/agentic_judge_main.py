"""Agentic LLM client implementation with native Gemini function calling using Python callables"""

import os
import json
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from google import genai
from utils.agent_utils import (
    create_agentic_unstructured_config,
    create_structuring_config, 
    extract_thought_summary, 
    is_agentic_model_supported,
    AGENTIC_SUPPORTED_MODELS,
    STRUCTURING_MODEL,
    function_call_logger
)
from llms.prompt_engineered_judge_main import LLMClient, TranslationJudgment

# Prompt template for converting unstructured agentic output to structured format
STRUCTURING_PROMPT_TEMPLATE = """You are a data extraction assistant. Your task is to convert the unstructured translation judgment below into a structured JSON format.

The unstructured judgment contains an evaluation of an English-to-Filipino translation based on 6 criteria. Extract the information and format it exactly as specified:

UNSTRUCTURED JUDGMENT:
{unstructured_content}

REQUIRED JSON FORMAT:
You must output a JSON object with exactly these fields:
- accuracy: boolean (true/false)
- accuracy_explanation: string
- fluency: boolean (true/false) 
- fluency_explanation: string
- coherence: boolean (true/false)
- coherence_explanation: string
- cultural_appropriateness: boolean (true/false)
- cultural_appropriateness_explanation: string
- guideline_adherence: boolean (true/false)
- guideline_adherence_explanation: string
- completeness: boolean (true/false)
- completeness_explanation: string

INSTRUCTIONS:
1. Carefully read the unstructured judgment
2. For each criterion, determine if it was marked as met (true) or not met (false)
3. Extract the explanation for each criterion
4. Output only valid JSON with no additional text
5. If any criterion is unclear, mark it as false and note the uncertainty in the explanation

Output the JSON now:"""

class AgenticGeminiClient(LLMClient):
    """
    Agentic Google Gemini API client with native function calling using Python callables.
    
    Features:
    - Google Search grounding for real-time information
    - Thought Summary capture for transparency
    - Native function calling with Python callable functions
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
        Generate structured translation judgment using native Gemini function calling with Python callables.
        
        Process:
        1. Agentic LLM call with Google Search, thought summaries, and Python callable function tools
        2. Native function call handling using Gemini's built-in capabilities
        3. Conversation management with function responses
        4. Structuring LLM call to convert final output to structured format
        
        Args:
            prompt (str): The evaluation prompt
            
        Returns:
            Dict containing structured judgment results and metadata
        """
        print(f"[DEBUG] AgenticGeminiClient.generate_judgment called with prompt length: {len(prompt)}")
        print(f"[DEBUG] Prompt preview: {prompt[:100]}...")
        
        try:
            # Clear any previous function call logs to start fresh
            function_call_logger.clear_logs()
            print("[DEBUG] Function call logger cleared for new evaluation")
            
            # Step 1: Native function calling conversation with Python callables
            print("[DEBUG] Step 1: Starting native function calling conversation with Python callables...")
            
            # Create agentic configuration with ONLY custom functions (no Google Search)
            # Google Search is handled by separate sub-agent calls in execute_search_expert
            agentic_config = create_agentic_unstructured_config(
                include_google_search=False,  # CRITICAL: No Google Search in main agent
                include_thoughts=True,
                include_custom_functions=True  # Only custom Python functions
            )
            
            print("[DEBUG] Agentic configuration created with Python callable tools")
            
            # Generate content with tools enabled - Gemini will handle function calls automatically
            agentic_response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=agentic_config
            )
            
            print("[DEBUG] Agentic response received, capturing function call logs...")
            
            # Capture function call logs after the agentic response
            captured_logs = function_call_logger.get_logs()
            print(f"[DEBUG] Captured {len(captured_logs)} function call logs")
            
            # Extract thought summary and main content from response
            extracted = extract_thought_summary(agentic_response)
            thought_summary = extracted.get("thought_summary", "") or ""
            unstructured_content = extracted.get("main_content", "") or ""
            has_thoughts = extracted.get("has_thoughts", False)
            
            print(f"[DEBUG] Response: Thought summary captured: {has_thoughts}")
            print(f"[DEBUG] Response: Content length: {len(unstructured_content)}")
            
            # Check if any function calls were made (this will be implicit with Python callables)
            function_calls_detected = any(term in unstructured_content.lower() for term in [
                "back-translation", "search", "research", "cultural", "semantic"
            ])
            
            print(f"[DEBUG] Function usage detected in response: {function_calls_detected}")
            
            # Step 2: Structure the unstructured output
            print("[DEBUG] Step 2: Converting unstructured output to structured format...")
            structured_result = self._convert_to_structured(unstructured_content)
            
            if not structured_result["success"]:
                print(f"[DEBUG] Step 2: Structuring failed: {structured_result.get('error', 'Unknown error')}")
                return {
                    "success": False,
                    "error": f"Structuring failed: {structured_result.get('error', 'Unknown error')}",
                    "raw_agentic_response": unstructured_content,
                    "thought_summary": thought_summary if has_thoughts else None,
                    "function_call_logs": captured_logs
                }
            
            print("[DEBUG] Step 2: Successfully converted to structured format")
            
            # Combine structured data with thought summary
            judgment_data = structured_result.get("data", {})
            
            return {
                "success": True,
                "data": judgment_data,
                "raw_agentic_response": unstructured_content,
                "raw_structuring_response": structured_result.get("raw_response", ""),
                "thought_summary": thought_summary if has_thoughts else None,
                "function_call_logs": captured_logs,
                "agentic_features": {
                    "google_search_enabled": True,  # Via sub-agent calls
                    "thought_summary_captured": has_thoughts,
                    "custom_functions_available": True,
                    "python_callable_functions": True,
                    "two_layer_architecture": True,  # Main agent + search sub-agent
                    "function_usage_detected": function_calls_detected,
                    "agentic_model": self.model_name,
                    "structuring_model": STRUCTURING_MODEL
                }
            }
            
        except Exception as e:
            print(f"[DEBUG] Agentic Gemini API error: {str(e)}")
            
            # Try to capture any function call logs even in error case
            try:
                captured_logs = function_call_logger.get_logs()
                print(f"[DEBUG] Captured {len(captured_logs)} function call logs during error")
            except:
                captured_logs = []
                print("[DEBUG] Could not capture function call logs during error")
            
            return {
                "success": False,
                "error": f"Agentic Gemini API error: {str(e)}",
                "function_call_logs": captured_logs
            }
    
    def _convert_to_structured(self, unstructured_content: str) -> Dict[str, Any]:
        """
        Convert unstructured agentic output to structured format.
        
        Args:
            unstructured_content (str): Unstructured output from agentic LLM
            
        Returns:
            Dict containing success status, structured data, and raw response
        """
        try:
            print(f"[DEBUG] Structuring: Converting {len(unstructured_content)} chars to structured format")
            
            # Create structuring prompt
            structuring_prompt = STRUCTURING_PROMPT_TEMPLATE.format(
                unstructured_content=unstructured_content
            )
            
            print(f"[DEBUG] Structuring: Prompt length: {len(structuring_prompt)}")
            
            # Create structuring configuration (structured output only, no tools)
            structuring_config = create_structuring_config(TranslationJudgment)
            
            # Call structuring LLM
            structuring_response = self.client.models.generate_content(
                model=STRUCTURING_MODEL,
                contents=structuring_prompt,
                config=structuring_config
            )
            
            print(f"[DEBUG] Structuring: Response received, length: {len(structuring_response.text)}")
            print(f"[DEBUG] Structuring: Full response: {structuring_response.text}")
            
            # Parse and validate structured response
            judgment_data = json.loads(structuring_response.text)
            judgment = TranslationJudgment(**judgment_data)
            
            print("[DEBUG] Structuring: Successfully validated with Pydantic")
            
            return {
                "success": True,
                "data": judgment.model_dump(),
                "raw_response": structuring_response.text
            }
            
        except json.JSONDecodeError as e:
            print(f"[DEBUG] Structuring: JSON decode error: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to parse structured JSON: {str(e)}",
                "raw_response": structuring_response.text if 'structuring_response' in locals() else None
            }
        except Exception as e:
            print(f"[DEBUG] Structuring: Error: {str(e)}")
            return {
                "success": False,
                "error": f"Structuring error: {str(e)}",
                "raw_response": getattr(locals().get('structuring_response'), 'text', None) or ""
            }

def create_agentic_llm_client(provider: str, model: str) -> Optional[LLMClient]:
    """
    Factory function to create agentic LLM client with Python callable functions.
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
    
    print("[DEBUG] Valid Google API key found, creating AgenticGeminiClient with Python callables")
    try:
        return AgenticGeminiClient(api_key, model)
    except Exception as e:
        print(f"[DEBUG] Error creating AgenticGeminiClient: {e}")
        raise e