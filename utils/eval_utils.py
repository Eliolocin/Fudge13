"""Utility functions for prompt processing and token replacement"""

import re
from typing import Dict, Any

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
