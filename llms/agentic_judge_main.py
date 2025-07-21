"""Agentic LLM client implementation with Google Search grounding and Thought Summaries"""

import os
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel
from google import genai
from utils.agent_utils import (
    create_agentic_unstructured_config,
    create_structuring_config, 
    extract_thought_summary, 
    is_agentic_model_supported,
    AGENTIC_SUPPORTED_MODELS,
    STRUCTURING_MODEL
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
        Generate structured translation judgment using two-step agentic process:
        1. Agentic LLM call with Google Search grounding and thought summaries (unstructured output)
        2. Structuring LLM call to convert unstructured output to structured format
        
        This approach works around Gemini API limitations where tools and structured output
        cannot be used simultaneously.
        """
        print(f"[DEBUG] AgenticGeminiClient.generate_judgment called with prompt length: {len(prompt)}")
        print(f"[DEBUG] Prompt preview: {prompt[:100]}...")
        print(f"[DEBUG] EXACT PROMPT SENT TO AGENTIC GEMINI: {repr(prompt)}")
        
        try:
            # Step 1: Agentic LLM call with tools (unstructured output)
            print("[DEBUG] Step 1: Calling Agentic Gemini API with Google Search grounding and thought summaries...")
            
            agentic_config = create_agentic_unstructured_config(
                include_google_search=True,
                include_thoughts=True
            )
            
            agentic_response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=agentic_config
            )
            
            print(f"[DEBUG] Step 1: Agentic Gemini API response received")
            
            # Extract thought summary and main content from agentic response
            extracted = extract_thought_summary(agentic_response)
            thought_summary = extracted["thought_summary"]
            unstructured_content = extracted["main_content"]
            has_thoughts = extracted["has_thoughts"]
            
            print(f"[DEBUG] Step 1: Thought summary extracted: {has_thoughts}")
            if has_thoughts:
                print(f"[DEBUG] Step 1: Thought summary length: {len(thought_summary)}")
                print(f"[DEBUG] Step 1: Thought summary preview: {thought_summary[:200]}...")
            
            print(f"[DEBUG] Step 1: Unstructured content length: {len(unstructured_content)}")
            print(f"[DEBUG] Step 1: Unstructured content preview: {unstructured_content[:300]}...")
            
            # Step 2: Structure the unstructured output
            print("[DEBUG] Step 2: Converting unstructured output to structured format...")
            structured_result = self._convert_to_structured(unstructured_content)
            
            if not structured_result["success"]:
                print(f"[DEBUG] Step 2: Structuring failed: {structured_result['error']}")
                return {
                    "success": False,
                    "error": f"Structuring failed: {structured_result['error']}",
                    "raw_agentic_response": unstructured_content,
                    "thought_summary": thought_summary if has_thoughts else None
                }
            
            print("[DEBUG] Step 2: Successfully converted to structured format")
            
            # Combine structured data with thought summary
            judgment_data = structured_result["data"]
            
            return {
                "success": True,
                "data": judgment_data,
                "raw_agentic_response": unstructured_content,
                "raw_structuring_response": structured_result["raw_response"],
                "thought_summary": thought_summary if has_thoughts else None,
                "agentic_features": {
                    "google_search_enabled": True,
                    "thought_summary_captured": has_thoughts,
                    "two_step_process": True,
                    "agentic_model": self.model_name,
                    "structuring_model": STRUCTURING_MODEL
                }
            }
            
        except Exception as e:
            print(f"[DEBUG] Agentic Gemini API error: {str(e)}")
            return {
                "success": False,
                "error": f"Agentic Gemini API error: {str(e)}"
            }
    
    def _convert_to_structured(self, unstructured_content: str) -> Dict[str, Any]:
        """
        Private method to convert unstructured agentic output to structured format.
        
        Uses a separate LLM call with structured output (no tools) to extract
        the 6 boolean criteria and explanations from the unstructured text.
        
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
                "error": f"Structuring error: {str(e)}"
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