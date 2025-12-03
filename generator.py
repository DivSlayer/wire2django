"""
Code generator for Django models, forms, and templates using Jinja2.
"""
import os
import logging
from typing import List, Dict
from jinja2 import Environment, FileSystemLoader, Template

logger = logging.getLogger(__name__)

# Set up Jinja2 environment
template_dir = os.path.join(os.path.dirname(__file__), 'templates_code')
env = Environment(
    loader=FileSystemLoader(template_dir),
    trim_blocks=True,
    lstrip_blocks=True
)


def generate_model_code(fields: List[Dict], model_name: str = "AutoModel") -> str:
    """
    Generate Django models.py code from detected fields.
    
    Args:
        fields: List of field dicts with keys: name, type
        model_name: Name of the Django model class
        
    Returns:
        Python code string for models.py
    """
    logger.info(f"Generating model code for {model_name} with {len(fields)} fields")
    
    template = env.get_template('model_template.jinja2')
    
    # Prepare field data for template
    field_list = []
    for field in fields:
        field_name = field.get('name', field.get('suggested_name', 'field'))
        field_type = field.get('type', field.get('suggested_type', 'CharField'))
        
        # Map Django field types to proper initialization
        max_len = get_max_length_for_field(field_type)
        field_dict = {
            'name': field_name,
            'type': field_type,
            'max_length': max_len if max_len is not None else None,
            'is_password': field_type == 'CharField' and 'password' in field_name.lower()
        }
        field_list.append(field_dict)
    
    code = template.render(model_name=model_name, fields=field_list)
    return code


def generate_form_code(fields: List[Dict], model_name: str = "AutoModel") -> str:
    """
    Generate Django forms.py code from detected fields.
    
    Args:
        fields: List of field dicts with keys: name, type
        model_name: Name of the Django model class
        
    Returns:
        Python code string for forms.py
    """
    logger.info(f"Generating form code for {model_name} with {len(fields)} fields")
    
    template = env.get_template('form_template.jinja2')
    
    # Prepare field data for template
    field_list = []
    for field in fields:
        field_name = field.get('name', field.get('suggested_name', 'field'))
        field_type = field.get('type', field.get('suggested_type', 'CharField'))
        
        field_dict = {
            'name': field_name,
            'type': field_type,
            'max_length': get_max_length_for_field(field_type),
            'is_password': field_type == 'CharField' and 'password' in field_name.lower()
        }
        field_list.append(field_dict)
    
    code = template.render(model_name=model_name, fields=field_list)
    return code


def generate_html_template(fields: List[Dict], model_name: str = "AutoModel") -> str:
    """
    Generate Django HTML template for list/create view.
    
    Args:
        fields: List of field dicts with keys: name, type
        model_name: Name of the Django model class
        
    Returns:
        HTML template code string
    """
    logger.info(f"Generating HTML template for {model_name} with {len(fields)} fields")
    
    template = env.get_template('html_template.jinja2')
    
    # Prepare field data for template
    field_list = []
    for field in fields:
        field_name = field.get('name', field.get('suggested_name', 'field'))
        field_dict = {
            'name': field_name,
            'label': field_name.replace('_', ' ').title()
        }
        field_list.append(field_dict)
    
    code = template.render(model_name=model_name, fields=field_list)
    return code


def get_max_length_for_field(field_type: str) -> int:
    """
    Return appropriate max_length for CharField based on field type.
    
    Args:
        field_type: Django field type string
        
    Returns:
        Suggested max_length value
    """
    if field_type == "EmailField":
        return 254  # Standard email max length
    elif field_type == "CharField":
        return 255  # Standard default
    elif field_type == "TextField":
        return None  # TextField doesn't have max_length
    else:
        return 255  # Default fallback


def generate(fields: List[Dict], model_name: str = "AutoModel") -> Dict[str, str]:
    """
    Generate all Django code artifacts (models, forms, template).
    
    Args:
        fields: List of field dicts with keys: name, type
        model_name: Name of the Django model class
        
    Returns:
        Dict mapping filenames to code strings:
            {
                'models.py': '...',
                'forms.py': '...',
                'list_create.html': '...'
            }
    """
    logger.info(f"Generating all Django artifacts for {model_name}")
    
    return {
        'models.py': generate_model_code(fields, model_name),
        'forms.py': generate_form_code(fields, model_name),
        'list_create.html': generate_html_template(fields, model_name)
    }

