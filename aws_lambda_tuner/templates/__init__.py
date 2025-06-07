"""
Templates package for AWS Lambda Tuner configuration templates.
"""

from pathlib import Path
import json
from typing import Dict, Any

# Template directory path
TEMPLATES_DIR = Path(__file__).parent


def load_template(template_name: str) -> Dict[str, Any]:
    """
    Load a configuration template by name.
    
    Args:
        template_name: Name of the template (without .json extension)
        
    Returns:
        Template configuration dictionary
    """
    template_path = TEMPLATES_DIR / f"{template_name}.json"
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_name}")
    
    with open(template_path, 'r') as f:
        return json.load(f)


def list_templates() -> list:
    """
    List all available templates.
    
    Returns:
        List of template names
    """
    templates = []
    for path in TEMPLATES_DIR.glob("*.json"):
        if path.stem != "__init__":
            templates.append(path.stem)
    return sorted(templates)
