"""LLM client implementations for translation judging"""

import os
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel
from google import genai
from openai import OpenAI


class TranslationJudgment(BaseModel):
    """Structured output schema for translation judgment - NO field descriptions to avoid prompt contamination"""
    accuracy: bool
    accuracy_explanation: str
    
    fluency: bool
    fluency_explanation: str
    
    coherence: bool
    coherence_explanation: str
    
    cultural_appropriateness: bool
    cultural_appropriateness_explanation: str
    
    guideline_adherence: bool
    guideline_adherence_explanation: str
    
    completeness: bool
    completeness_explanation: str
    
    # Optional field for agentic mode - captures LLM's reasoning process
    thought_summary: Optional[str] = None


class LLMClient:
    """Base class for LLM clients"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def generate_judgment(self, prompt: str) -> Dict[str, Any]:
        """Generate structured translation judgment"""
        raise NotImplementedError("Subclasses must implement this method")


class GeminiClient(LLMClient):
    """Google Gemini API client with structured output using new google-genai package"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        super().__init__(api_key)
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
    
    def generate_judgment(self, prompt: str) -> Dict[str, Any]:
        """Generate structured translation judgment using Gemini"""
        print(f"[DEBUG] GeminiClient.generate_judgment called with prompt length: {len(prompt)}")
        print(f"[DEBUG] Prompt preview: {prompt[:100]}...")
        print(f"[DEBUG] EXACT PROMPT SENT TO GEMINI: {repr(prompt)}")
        
        try:
            print("[DEBUG] Calling Gemini API with structured output...")
            
            # Generate content with structured output using new API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": TranslationJudgment,
                }
            )
            
            print(f"[DEBUG] Gemini API response received, length: {len(response.text)}")
            print(f"[DEBUG] Full raw response: {response.text}")
            
            # Parse the structured response
            judgment_data = json.loads(response.text)
            print("[DEBUG] Successfully parsed JSON response")
            
            # Validate with Pydantic
            judgment = TranslationJudgment(**judgment_data)
            print("[DEBUG] Successfully validated with Pydantic")
            
            return {
                "success": True,
                "data": judgment.model_dump(),
                "raw_response": response.text
            }
            
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON decode error: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to parse JSON response: {str(e)}",
                "raw_response": response.text if 'response' in locals() else None
            }
        except Exception as e:
            print(f"[DEBUG] Gemini API error: {str(e)}")
            return {
                "success": False,
                "error": f"Gemini API error: {str(e)}"
            }


class OpenAIClient(LLMClient):
    """OpenAI API client with structured output"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        super().__init__(api_key)
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
    
    def generate_judgment(self, prompt: str) -> Dict[str, Any]:
        """Generate structured translation judgment using OpenAI"""
        print(f"[DEBUG] OpenAIClient.generate_judgment called with prompt length: {len(prompt)}")
        print(f"[DEBUG] EXACT PROMPT SENT TO OPENAI: {repr(prompt)}")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}  # ONLY the pure system prompt
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "translation_judgment",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "accuracy": {"type": "boolean"},
                                "accuracy_explanation": {"type": "string"},
                                "fluency": {"type": "boolean"},
                                "fluency_explanation": {"type": "string"},
                                "coherence": {"type": "boolean"},
                                "coherence_explanation": {"type": "string"},
                                "cultural_appropriateness": {"type": "boolean"},
                                "cultural_appropriateness_explanation": {"type": "string"},
                                "guideline_adherence": {"type": "boolean"},
                                "guideline_adherence_explanation": {"type": "string"},
                                "completeness": {"type": "boolean"},
                                "completeness_explanation": {"type": "string"}
                            },
                            "required": [
                                "accuracy", "accuracy_explanation",
                                "fluency", "fluency_explanation", 
                                "coherence", "coherence_explanation",
                                "cultural_appropriateness", "cultural_appropriateness_explanation",
                                "guideline_adherence", "guideline_adherence_explanation",
                                "completeness", "completeness_explanation"
                            ],
                            "additionalProperties": False
                        }
                    }
                }
            )
            
            response_text = response.choices[0].message.content
            judgment_data = json.loads(response_text)
            
            # Validate with Pydantic
            judgment = TranslationJudgment(**judgment_data)
            
            return {
                "success": True,
                "data": judgment.model_dump(),
                "raw_response": response_text
            }
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Failed to parse JSON response: {str(e)}",
                "raw_response": response_text if 'response_text' in locals() else None
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"OpenAI API error: {str(e)}"
            }


def create_llm_client(provider: str, model: str) -> Optional[LLMClient]:
    """Factory function to create appropriate LLM client"""
    import os
    from dotenv import load_dotenv
    
    # Reload environment variables
    load_dotenv()
    
    print(f"[DEBUG] Creating client for provider='{provider}', model='{model}'")
    
    if provider == "demo":
        print("[DEBUG] Demo provider selected, returning DemoClient")
        return DemoClient(provider, model)
    
    elif provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        print(f"[DEBUG] Google provider - API key exists: {bool(api_key)}")
        if api_key:
            print(f"[DEBUG] API key length: {len(api_key)}, starts with: {api_key[:15]}...")
        
        if not api_key or api_key.startswith("demo_key"):
            print("[DEBUG] No valid Google API key, using DemoClient")
            return DemoClient(provider, model)
        
        print("[DEBUG] Valid Google API key found, creating GeminiClient")
        try:
            return GeminiClient(api_key, model)
        except Exception as e:
            print(f"[DEBUG] Error creating GeminiClient: {e}")
            return DemoClient(provider, model)
    
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        print(f"[DEBUG] OpenAI provider - API key exists: {bool(api_key)}")
        if api_key:
            print(f"[DEBUG] API key length: {len(api_key)}, starts with: {api_key[:15]}...")
            
        if not api_key or api_key.startswith("demo_key"):
            print("[DEBUG] No valid OpenAI API key, using DemoClient")
            return DemoClient(provider, model)
            
        print("[DEBUG] Valid OpenAI API key found, creating OpenAIClient")
        try:
            return OpenAIClient(api_key, model)
        except Exception as e:
            print(f"[DEBUG] Error creating OpenAIClient: {e}")
            return DemoClient(provider, model)
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")


class DemoClient(LLMClient):
    """Demo client for testing without real API keys"""
    
    def __init__(self, provider: str, model: str):
        super().__init__("demo_key")
        self.provider = provider
        self.model = model
    
    def generate_judgment(self, prompt: str) -> Dict[str, Any]:
        """Generate demo structured translation judgment"""
        import time
        import random
        
        print(f"[DEBUG] DemoClient.generate_judgment called with prompt length: {len(prompt)}")
        
        # Simulate API delay
        time.sleep(1)
        
        # Generate realistic demo data
        criteria_scores = [random.choice([True, False]) for _ in range(6)]
        
        demo_judgment = {
            "accuracy": criteria_scores[0],
            "accuracy_explanation": "The translation accurately conveys the core meaning of the English source text with appropriate Filipino vocabulary and structure.",
            
            "fluency": criteria_scores[1],
            "fluency_explanation": "The Filipino translation reads naturally and fluently, following proper grammatical structures and word order typical of the language.",
            
            "coherence": criteria_scores[2],
            "coherence_explanation": "The translation maintains logical flow and coherence, with clear connections between ideas and proper discourse markers.",
            
            "cultural_appropriateness": criteria_scores[3],
            "cultural_appropriateness_explanation": "The translation appropriately adapts cultural references and context to Filipino culture while maintaining the intent of the original.",
            
            "guideline_adherence": criteria_scores[4],
            "guideline_adherence_explanation": "The translation follows established Filipino translation guidelines including proper register, terminology, and stylistic conventions.",
            
            "completeness": criteria_scores[5],
            "completeness_explanation": "The translation is complete without significant omissions or unnecessary additions that would alter the original meaning."
        }
        
        return {
            "success": True,
            "data": demo_judgment,
            "raw_response": f"Demo response from {self.provider} {self.model}"
        }
