"""Utility functions for prompt processing and token replacement"""

import re
import random


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


def randomize_positions(prompt):
    """Randomize the order of content within <pos></pos> tags to reduce position bias
    
    This function finds all <pos>...</pos> blocks in the prompt, extracts their content,
    shuffles the order randomly, and replaces the original blocks with shuffled content.
    The <pos> tags are removed from the final output.
    
    Args:
        prompt (str): The prompt containing <pos></pos> tags around content to randomize
        
    Returns:
        str: The prompt with randomized position content and <pos> tags removed
        
    Example:
        Input: "Rate these criteria:\n<pos>1. Accuracy: ...</pos>\n<pos>2. Fluency: ...</pos>"
        Output: "Rate these criteria:\n2. Fluency: ...\n1. Accuracy: ..." (order randomized)
    """
    
    # Find all <pos>...</pos> blocks using regex with DOTALL for multiline content
    pattern = r'<pos>(.*?)</pos>'
    matches = re.findall(pattern, prompt, re.DOTALL)
    
    # If no <pos> tags found, return original prompt unchanged
    if not matches:
        return prompt
    
    # Create shuffled copy of the matches
    shuffled_matches = matches.copy()
    random.shuffle(shuffled_matches)
    
    # Replace each <pos>...</pos> block with shuffled content
    result = prompt
    for i, original_content in enumerate(matches):
        # Find and replace the i-th <pos> block with the shuffled content
        old_block = f'<pos>{original_content}</pos>'
        new_content = shuffled_matches[i]
        # Replace only the first occurrence to maintain order
        result = result.replace(old_block, new_content.strip(), 1)
    
    return result


