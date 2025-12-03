"""
Utility functions for name sanitization and field name processing.
"""
import re
import keyword
import logging

logger = logging.getLogger(__name__)


def slugify_name(text: str) -> str:
    """
    Convert text to a slug-like identifier.
    
    Args:
        text: Input text to slugify
        
    Returns:
        Slugified string with lowercase, spaces to underscores, non-alphanumeric removed
    """
    if not text:
        return ""
    
    # Lowercase and replace spaces/underscores with single underscore
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', '_', text)  # Replace whitespace with underscore
    text = re.sub(r'_+', '_', text)  # Collapse multiple underscores
    
    return text


def sanitize_identifier(name: str, fallback_prefix: str = "field") -> str:
    """
    Sanitize a field name to be a valid Python identifier.
    
    Rules:
    - Convert to snake_case
    - Only allow [a-z0-9_]+
    - No leading digits
    - Must not be a Python reserved word
    - If empty or invalid, return fallback
    
    Args:
        name: Raw field name to sanitize
        fallback_prefix: Prefix for fallback names (default: "field")
        
    Returns:
        Valid Python identifier
    """
    if not name:
        return f"{fallback_prefix}_1"
    
    # First slugify
    sanitized = slugify_name(name)
    
    # Remove any remaining non-alphanumeric except underscores
    sanitized = re.sub(r'[^a-z0-9_]', '', sanitized)
    
    # Remove leading digits or underscores
    sanitized = re.sub(r'^[\d_]+', '', sanitized)
    
    # If empty after cleaning, use fallback
    if not sanitized:
        return f"{fallback_prefix}_1"
    
    # Check if it's a Python reserved word
    if sanitized in keyword.kwlist:
        sanitized = f"{sanitized}_field"
    
    # Ensure it doesn't start with a digit
    if sanitized and sanitized[0].isdigit():
        sanitized = f"{fallback_prefix}_{sanitized}"
    
    return sanitized


def ensure_unique_field_names(fields: list) -> list:
    """
    Ensure all field names in a list are unique by appending numbers if needed.
    
    Args:
        fields: List of dicts with 'name' or 'suggested_name' key
        
    Returns:
        List of fields with unique names
    """
    seen = set()
    result = []
    
    for field in fields:
        name = field.get('name') or field.get('suggested_name', 'field')
        original_name = name
        
        counter = 1
        while name in seen:
            name = f"{original_name}_{counter}"
            counter += 1
        
        seen.add(name)
        field = field.copy()
        field['name'] = name
        result.append(field)
    
    return result

