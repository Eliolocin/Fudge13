"""Agent utilities for agentic LLM functionality with native Gemini function calling"""

from google import genai
from google.genai import types
from typing import List, Dict, Any, Optional
import os
import logging
import json
import time
from datetime import datetime
from threading import Lock

# Translation libraries - will be imported conditionally
try:
    from deep_translator import GoogleTranslator
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    logging.warning("Translation libraries not available. Install deep-translator for Back-Translation Validator functionality.")

class FunctionCallLogger:
    """
    Thread-safe logger for capturing detailed function call information in agentic systems.
    Provides comprehensive logging of function calls, API interactions, and processing results.
    """
    
    def __init__(self):
        """Initialize the function call logger with thread safety."""
        self._logs = []
        self._lock = Lock()
    
    def start_function_call(self, function_name: str, parameters: Dict[str, Any]) -> str:
        """
        Start logging a function call and return a unique call_id.
        
        Args:
            function_name (str): Name of the function being called
            parameters (Dict[str, Any]): Parameters passed to the function
            
        Returns:
            str: Unique call_id for tracking this function call
        """
        call_id = f"{function_name}_{int(time.time() * 1000)}"
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "call_id": call_id,
            "timestamp": timestamp,
            "function_name": function_name,
            "parameters": parameters.copy(),
            "start_time": time.time(),
            "api_calls": [],
            "processed_result": {},
            "execution_time_ms": None,
            "success": None,
            "error": None,
            "return_value": None
        }
        
        with self._lock:
            self._logs.append(log_entry)
        
        return call_id
    
    def log_api_call(self, call_id: str, service: str, request: Any, response: Any, execution_time_ms: Optional[float] = None):
        """
        Log an API call made within a function.
        
        Args:
            call_id (str): The function call ID this API call belongs to
            service (str): Name of the service called (e.g., 'google_translate', 'google_search')
            request (Any): The request sent to the API
            response (Any): The response received from the API
            execution_time_ms (Optional[float]): Time taken for this API call
        """
        api_call_entry = {
            "service": service,
            "request": str(request)[:500] if request else None,  # Truncate long requests
            "response": str(response)[:1000] if response else None,  # Truncate long responses
            "execution_time_ms": execution_time_ms,
            "timestamp": datetime.now().isoformat()
        }
        
        with self._lock:
            for log_entry in self._logs:
                if log_entry["call_id"] == call_id:
                    log_entry["api_calls"].append(api_call_entry)
                    break
    
    def log_sub_agent_call(self, call_id: str, agent_type: str, sub_response: Any, execution_time_ms: Optional[float] = None):
        """
        Log a sub-agent call made within a function.
        
        Args:
            call_id (str): The function call ID this sub-agent call belongs to
            agent_type (str): Type of sub-agent called (e.g., 'search_only')
            sub_response (Any): The response from the sub-agent
            execution_time_ms (Optional[float]): Time taken for this sub-agent call
        """
        sub_agent_entry = {
            "agent_type": agent_type,
            "response": str(sub_response)[:1000] if sub_response else None,
            "execution_time_ms": execution_time_ms,
            "timestamp": datetime.now().isoformat()
        }
        
        with self._lock:
            for log_entry in self._logs:
                if log_entry["call_id"] == call_id:
                    if "sub_agent_calls" not in log_entry:
                        log_entry["sub_agent_calls"] = []
                    log_entry["sub_agent_calls"].append(sub_agent_entry)
                    break
    
    def log_processing_result(self, call_id: str, processed_result: Dict[str, Any]):
        """
        Log the processed result of a function call.
        
        Args:
            call_id (str): The function call ID
            processed_result (Dict[str, Any]): The processed result data
        """
        with self._lock:
            for log_entry in self._logs:
                if log_entry["call_id"] == call_id:
                    log_entry["processed_result"] = processed_result.copy()
                    break
    
    def end_function_call(self, call_id: str, success: bool, return_value: Any, error: Optional[str] = None):
        """
        End logging a function call with results.
        
        Args:
            call_id (str): The function call ID
            success (bool): Whether the function call succeeded
            return_value (Any): The value returned by the function
            error (Optional[str]): Error message if the call failed
        """
        end_time = time.time()
        
        with self._lock:
            for log_entry in self._logs:
                if log_entry["call_id"] == call_id:
                    execution_time_ms = (end_time - log_entry["start_time"]) * 1000
                    log_entry["execution_time_ms"] = round(execution_time_ms, 2)
                    log_entry["success"] = success
                    log_entry["error"] = error
                    log_entry["return_value"] = str(return_value)[:1000] if return_value else None
                    break
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """
        Get all logged function calls.
        
        Returns:
            List[Dict[str, Any]]: List of all function call logs
        """
        with self._lock:
            # Return clean copies without internal timing data
            clean_logs = []
            for log in self._logs:
                clean_log = log.copy()
                # Remove internal timing data
                clean_log.pop("start_time", None)
                clean_logs.append(clean_log)
            return clean_logs
    
    def clear_logs(self):
        """Clear all logged function calls."""
        with self._lock:
            self._logs.clear()

# Global function call logger instance
function_call_logger = FunctionCallLogger()

def create_google_search_tool() -> types.Tool:
    """
    Create Google Search grounding tool for enhanced translation evaluation.
    
    Returns:
        types.Tool: Configured Google Search grounding tool
    """
    return types.Tool(
        google_search=types.GoogleSearch()
    )

def create_thinking_config(include_thoughts: bool = True) -> types.ThinkingConfig:
    """
    Create thinking configuration for capturing thought summaries.
    
    Args:
        include_thoughts (bool): Whether to include thought summaries in response
        
    Returns:
        types.ThinkingConfig: Configuration for thought summary capture
    """
    return types.ThinkingConfig(
        include_thoughts=include_thoughts
    )

# Function implementations that will be called by Gemini
def execute_back_translation(english_text: str, filipino_translation: str) -> Dict[str, Any]:
    """
    Validate translation quality through back-translation analysis. 
    Translates Filipino text back to English using Google Translate and compares 
    with original for semantic preservation assessment. Use this when you need to 
    verify if the Filipino translation maintains the original meaning and semantic 
    accuracy. Particularly useful for detecting mistranslations or semantic drift.
    
    Args:
        english_text (str): The original English text to compare against
        filipino_translation (str): The Filipino translation to validate through back-translation
        
    Returns:
        Dict containing back-translation results and analysis
    """
    # Start function call logging
    call_id = function_call_logger.start_function_call(
        "execute_back_translation",
        {
            "english_text": english_text,
            "filipino_translation": filipino_translation
        }
    )
    
    if not TRANSLATION_AVAILABLE:
        error_result = {
            "success": False,
            "error": "Translation libraries not available. Install deep-translator for back-translation functionality.",
            "tool_used": "back_translation_validator"
        }
        function_call_logger.end_function_call(call_id, False, error_result, "Translation libraries not available")
        return error_result
    
    try:
        # Translate Filipino back to English using Google Translate
        api_start_time = time.time()
        translator = GoogleTranslator(source='tl', target='en')
        back_translated = translator.translate(filipino_translation)
        api_end_time = time.time()
        
        # Log the Google Translate API call
        function_call_logger.log_api_call(
            call_id,
            "google_translate",
            f"Translate '{filipino_translation}' from Filipino to English",
            f"Result: '{back_translated}'",
            round((api_end_time - api_start_time) * 1000, 2)
        )
        
        # Calculate basic similarity metrics
        original_words = set(english_text.lower().split())
        back_translated_words = set(back_translated.lower().split())
        
        # Simple word overlap calculation
        common_words = original_words.intersection(back_translated_words)
        word_overlap_ratio = len(common_words) / max(len(original_words), len(back_translated_words)) if max(len(original_words), len(back_translated_words)) > 0 else 0
        
        # Length comparison
        length_ratio = min(len(english_text), len(back_translated)) / max(len(english_text), len(back_translated)) if max(len(english_text), len(back_translated)) > 0 else 0
        
        # Prepare analysis results
        analysis_result = {
            "word_overlap_ratio": round(word_overlap_ratio, 3),
            "length_preservation_ratio": round(length_ratio, 3),
            "common_words": list(common_words),
            "semantic_similarity_indicator": "high" if word_overlap_ratio > 0.6 else "medium" if word_overlap_ratio > 0.3 else "low"
        }
        
        interpretation_result = {
            "semantic_preservation": "good" if word_overlap_ratio > 0.5 and length_ratio > 0.7 else "moderate" if word_overlap_ratio > 0.3 else "poor",
            "recommendation": "Translation preserves meaning well" if word_overlap_ratio > 0.5 else "Consider checking for semantic accuracy"
        }
        
        # Log the processing results
        processed_result = {
            "back_translated_text": back_translated,
            "semantic_similarity": analysis_result["semantic_similarity_indicator"],
            "preservation_quality": interpretation_result["semantic_preservation"],
            "word_overlap_ratio": analysis_result["word_overlap_ratio"],
            "length_preservation_ratio": analysis_result["length_preservation_ratio"],
            "common_words_count": len(common_words),
            "recommendation": interpretation_result["recommendation"]
        }
        function_call_logger.log_processing_result(call_id, processed_result)
        
        # Prepare final return value
        result = {
            "success": True,
            "tool_used": "back_translation_validator",
            "original_english": english_text,
            "filipino_translation": filipino_translation,
            "back_translated_english": back_translated,
            "analysis": analysis_result,
            "interpretation": interpretation_result
        }
        
        # End function call logging with success
        function_call_logger.end_function_call(call_id, True, result)
        return result
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": f"Back-translation failed: {str(e)}",
            "tool_used": "back_translation_validator"
        }
        function_call_logger.end_function_call(call_id, False, error_result, str(e))
        return error_result

def execute_search_expert(search_query: str, evaluation_context: str) -> Dict[str, Any]:
    """
    Research comprehensive cultural context, linguistic patterns, and translation guidelines. 
    Use this proactively when the English text contains: cultural references, media quotes, 
    pop culture (memes, TV shows, movies), literary content, idioms, slang, technical terms, 
    historical references, or any content that might benefit from cultural/contextual research 
    about Filipino culture and current usage patterns.
    
    Args:
        search_query (str): Specific search query to research (e.g., 'Game of Thrones Filipino translation memes', 'Winter is coming cultural meaning Philippines', 'Star Wars quotes Filipino pop culture')
        evaluation_context (str): Context about what we're evaluating and why this search is relevant
        
    Returns:
        Dict containing search results and analysis
    """
    # Start function call logging
    call_id = function_call_logger.start_function_call(
        "execute_search_expert",
        {
            "search_query": search_query,
            "evaluation_context": evaluation_context
        }
    )
    
    try:
        # Get API key
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            error_result = {
                "success": False,
                "error": "Google API key not available for search research",
                "tool_used": "search_expert"
            }
            function_call_logger.end_function_call(call_id, False, error_result, "Google API key not available")
            return error_result
        
        # Create a separate Gemini client for search-grounded research (sub-agent pattern)
        search_client = genai.Client(api_key=api_key)
        
        # Enhanced research prompt for comprehensive cultural investigation
        research_prompt = f"""You are a Filipino cultural and linguistic research expert helping evaluate English-to-Filipino translations with deep cultural awareness.

CONTEXT: {evaluation_context}
RESEARCH QUERY: {search_query}

Please conduct COMPREHENSIVE research covering these areas:

## PRIMARY RESEARCH AREAS:
1. **Cultural References & Context**: 
   - Historical, political, or social references
   - Religious or spiritual contexts in Filipino culture
   - Regional variations across Philippines (Luzon, Visayas, Mindanao)

2. **Pop Culture & Media Analysis**:
   - Filipino movies, TV shows, music that reference similar content
   - Memes, social media trends, viral content
   - Celebrity culture, entertainment industry context
   - Online communities and fandoms

3. **Current Filipino Usage Patterns**:
   - How modern Filipinos actually use/translate such expressions
   - Generational differences (Gen Z, Millennials, older generations)
   - Social media language, texting, informal communication
   - Code-switching patterns (Taglish usage)

4. **Translation Standards & Guidelines**:
   - Official Filipino translation guidelines
   - Academic/scholarly translation practices
   - Media dubbing/subtitling standards
   - Government and educational institution practices

5. **Linguistic & Semantic Analysis**:
   - Equivalent Filipino idioms or expressions
   - Loan words vs. native translations debate
   - Semantic fields and connotations in Filipino
   - Register appropriateness (formal vs. informal)

## OUTPUT REQUIREMENTS:
- Include specific examples from Filipino media/culture when found
- Note regional variations if applicable  
- Mention generational usage differences
- Cite current trends or recent developments
- Provide alternative translation options if better ones exist

Be thorough, specific, and culturally nuanced in your research."""
        
        # Create sub-agent configuration with Google Search only (no custom functions)
        search_config = create_search_only_config(include_thoughts=True)
        
        # Execute search-grounded research using sub-agent pattern
        sub_agent_start_time = time.time()
        search_response = search_client.models.generate_content(
            model="gemini-2.5-flash",  # Use flash for cost-effectiveness
            contents=research_prompt,
            config=search_config
        )
        sub_agent_end_time = time.time()
        
        # Extract research results
        research_content = search_response.text or ""
        
        # Log the sub-agent call
        function_call_logger.log_sub_agent_call(
            call_id,
            "search_only",
            {
                "model": "gemini-2.5-flash",
                "search_grounded": True,
                "response_length": len(research_content),
                "response_preview": research_content[:200] + "..." if len(research_content) > 200 else research_content
            },
            round((sub_agent_end_time - sub_agent_start_time) * 1000, 2)
        )
        
        # Enhanced analysis of research findings
        content_lower = research_content.lower()
        insights = {
            "cultural_context_found": any(term in content_lower for term in ["cultural", "tradition", "filipino culture", "philippines"]),
            "pop_culture_found": any(term in content_lower for term in ["meme", "viral", "trending", "social media", "tiktok", "facebook"]),
            "media_references_found": any(term in content_lower for term in ["movie", "tv show", "film", "series", "entertainment"]),
            "translation_guidelines_found": any(term in content_lower for term in ["guideline", "standard", "translation practice", "academic"]),
            "linguistic_patterns_found": any(term in content_lower for term in ["pattern", "filipino", "tagalog", "taglish", "code-switching"]),
            "regional_variations_found": any(term in content_lower for term in ["luzon", "visayas", "mindanao", "regional", "dialect"]),
            "generational_differences_found": any(term in content_lower for term in ["gen z", "millennial", "generation", "young", "older"]),
            "current_usage_found": any(term in content_lower for term in ["current", "modern", "recent", "today", "nowadays"]),
            "comprehensive_research": len(research_content) > 800
        }
        
        cultural_depth_score = sum([1 for v in insights.values() if v]) / len(insights)
        
        # Log the processing results
        processed_result = {
            "research_length": len(research_content),
            "cultural_depth_score": round(cultural_depth_score, 3),
            "insights_found": insights,
            "research_areas_covered": sum([1 for v in insights.values() if v]),
            "comprehensive_research": insights["comprehensive_research"],
            "key_findings_preview": research_content[:300] + "..." if len(research_content) > 300 else research_content
        }
        function_call_logger.log_processing_result(call_id, processed_result)
        
        # Prepare final return value
        result = {
            "success": True,
            "tool_used": "search_expert",
            "search_query": search_query,
            "evaluation_context": evaluation_context,
            "research_findings": research_content,
            "insights": insights,
            "cultural_depth_score": cultural_depth_score,
            "recommendation": "Use these comprehensive research findings to evaluate cultural appropriateness, current usage patterns, and alignment with Filipino cultural context. Pay special attention to pop culture references and modern usage patterns that may not be in training data."
        }
        
        # End function call logging with success
        function_call_logger.end_function_call(call_id, True, result)
        return result
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": f"Search expert research failed: {str(e)}",
            "tool_used": "search_expert"
        }
        function_call_logger.end_function_call(call_id, False, error_result, str(e))
        return error_result

def create_agentic_unstructured_config(
    include_google_search: bool = True,
    include_thoughts: bool = True,
    include_custom_functions: bool = True
) -> types.GenerateContentConfig:
    """
    Create configuration for agentic LLM behavior with tools but no structured output.
    
    IMPORTANT: Due to Gemini API limitations, Google Search and custom functions 
    cannot be used together. This config supports ONLY custom functions.
    Use create_search_only_config() for Google Search only.
    
    Args:
        include_google_search (bool): IGNORED - custom functions only
        include_thoughts (bool): Enable thought summary capture
        include_custom_functions (bool): Enable custom function tools
        
    Returns:
        types.GenerateContentConfig: Agentic configuration with custom functions only
    """
    tools = []
    
    # CRITICAL: Only add custom functions, NO Google Search
    # Google Search is handled by separate sub-agent calls
    if include_custom_functions:
        tools.extend([execute_back_translation, execute_search_expert])
    
    config_params = {}
    
    # Add tools if any are configured
    if tools:
        config_params['tools'] = tools
    
    # Add thinking configuration if enabled
    if include_thoughts:
        config_params['thinking_config'] = create_thinking_config(include_thoughts)
    
    return types.GenerateContentConfig(**config_params)

def create_search_only_config(
    include_thoughts: bool = True
) -> types.GenerateContentConfig:
    """
    Create configuration for Google Search only (no custom functions).
    
    This is used for the search sub-agent that handles Google Search grounding
    separately from the main agentic judge.
    
    Args:
        include_thoughts (bool): Enable thought summary capture
        
    Returns:
        types.GenerateContentConfig: Configuration with Google Search only
    """
    tools = [create_google_search_tool()]  # Only Google Search
    
    config_params = {'tools': tools}
    
    # Add thinking configuration if enabled
    if include_thoughts:
        config_params['thinking_config'] = create_thinking_config(include_thoughts)
    
    return types.GenerateContentConfig(**config_params)

def create_structuring_config(response_schema: Any) -> types.GenerateContentConfig:
    """
    Create configuration for structured output conversion.
    
    Args:
        response_schema (Any): Pydantic schema for response validation
        
    Returns:
        types.GenerateContentConfig: Configuration for structured output only
    """
    return types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_schema
    )

def extract_thought_summary(response) -> Dict[str, Any]:
    """
    Extract thought summary and main content from agentic response.
    
    Args:
        response: Response object from Gemini API
        
    Returns:
        Dict containing thought summary, main content, and has_thoughts flag
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

# Model support checking
AGENTIC_SUPPORTED_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-05-20"
]

STRUCTURING_MODEL = "gemini-2.5-flash"

def is_agentic_model_supported(model_name: str) -> bool:
    """
    Check if a model supports agentic features.
    
    Args:
        model_name (str): Name of the model to check
        
    Returns:
        bool: True if model supports agentic features
    """
    return model_name in AGENTIC_SUPPORTED_MODELS