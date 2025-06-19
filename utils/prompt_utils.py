"""Utility functions for prompt processing and token replacement"""

import re
from typing import Dict, Any


def replace_tokens(prompt, **kwargs):
    """Replace tokens in prompt with provided values"""
    
    # Available tokens
    tokens = {
        'source_text': kwargs.get('source_text', ''),
        'fil_translation': kwargs.get('fil_translation', ''),
        'ref_translation': kwargs.get('ref_translation', '')
    }
    
    # Handle optional reference translation
    if tokens['ref_translation'] and tokens['ref_translation'].strip():
        tokens['ref_translation'] = f"**Reference Translation (for comparison):**\n{tokens['ref_translation']}\n"
    else:
        tokens['ref_translation'] = ""
    
    # Replace each token
    result = prompt
    for token, value in tokens.items():
        pattern = f'{{{{{token}}}}}'
        result = result.replace(pattern, str(value))
    
    # Clean up any extra whitespace from empty ref_translation
    result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
    
    return result


def validate_tokens(prompt, required_tokens=None):
    """Validate that prompt contains required tokens"""
    
    if required_tokens is None:
        required_tokens = ['source_text', 'fil_translation']
    
    missing_tokens = []
    
    for token in required_tokens:
        pattern = f'{{{{{token}}}}}'
        if pattern not in prompt:
            missing_tokens.append(token)
    
    return {
        'valid': len(missing_tokens) == 0,
        'missing_tokens': missing_tokens
    }


def find_all_tokens(prompt):
    """Find all tokens in the prompt"""
    
    pattern = r'\{\{([^}]+)\}\}'
    matches = re.findall(pattern, prompt)
    return list(set(matches))


def calculate_final_score(judgment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate final score based on the six boolean criteria"""
    
    # Count true values for the six criteria
    criteria = [
        judgment_data.get("accuracy", False),
        judgment_data.get("fluency", False),
        judgment_data.get("coherence", False),
        judgment_data.get("cultural_appropriateness", False),
        judgment_data.get("guideline_adherence", False),
        judgment_data.get("completeness", False)
    ]
    
    true_count = sum(criteria)
    
    # Determine score and label based on true count
    if true_count >= 5:
        score = 5
        label = "Excellent"
        color = "success"
    elif true_count >= 3:
        score = 4 if true_count == 4 else 3
        label = "Good"
        color = "warning"
    else:
        score = 2 if true_count == 2 else 1
        label = "Poor"
        color = "error"
    
    return {
        "score": score,
        "label": label,
        "color": color,
        "true_count": true_count,
        "total_criteria": 6
    }
