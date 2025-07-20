"""Utility functions for prompt processing and token replacement"""

import re


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


