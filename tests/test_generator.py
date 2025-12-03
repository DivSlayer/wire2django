"""
Unit tests for generator.py code generation functionality.
"""
import os
import sys
import ast

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from generator import generate_model_code, generate_form_code, generate_html_template, generate


def test_generate_model_code():
    """Test Django model code generation."""
    fields = [
        {'name': 'email', 'type': 'EmailField'},
        {'name': 'full_name', 'type': 'CharField'},
        {'name': 'age', 'type': 'IntegerField'},
    ]
    
    code = generate_model_code(fields, 'UserModel')
    
    # Verify it's valid Python
    try:
        ast.parse(code)
    except SyntaxError as e:
        pytest.fail(f"Generated code is not valid Python: {e}")
    
    # Verify key components
    assert 'from django.db import models' in code
    assert 'class UserModel' in code
    assert 'email = models.EmailField' in code
    assert 'full_name = models.CharField' in code
    assert 'age = models.IntegerField' in code
    assert 'def __str__' in code


def test_generate_form_code():
    """Test Django form code generation."""
    fields = [
        {'name': 'email', 'type': 'EmailField'},
        {'name': 'password', 'type': 'CharField'},
    ]
    
    code = generate_form_code(fields, 'LoginFormModel')
    
    # Verify it's valid Python
    try:
        ast.parse(code)
    except SyntaxError as e:
        pytest.fail(f"Generated code is not valid Python: {e}")
    
    # Verify key components
    assert 'from django import forms' in code
    assert 'class LoginFormModelForm' in code
    assert 'model = LoginFormModel' in code
    assert 'widgets' in code


def test_generate_html_template():
    """Test HTML template generation."""
    fields = [
        {'name': 'email', 'type': 'EmailField'},
        {'name': 'name', 'type': 'CharField'},
    ]
    
    html = generate_html_template(fields, 'ContactModel')
    
    # Verify key HTML components
    assert '<!DOCTYPE html>' in html
    assert 'ContactModel' in html
    assert '{{ form.as_p }}' in html
    assert 'email' in html.lower()
    assert '<table>' in html


def test_generate_all_artifacts():
    """Test complete code generation for all artifacts."""
    fields = [
        {'name': 'email', 'type': 'EmailField', 'suggested_name': 'email', 'suggested_type': 'EmailField'},
        {'name': 'first_name', 'type': 'CharField', 'suggested_name': 'first_name', 'suggested_type': 'CharField'},
    ]
    
    artifacts = generate(fields, 'ContactModel')
    
    # Verify all artifacts are generated
    assert 'models.py' in artifacts
    assert 'forms.py' in artifacts
    assert 'list_create.html' in artifacts
    
    # Verify models.py content
    assert 'class ContactModel' in artifacts['models.py']
    assert 'email = models.EmailField' in artifacts['models.py']
    
    # Verify forms.py content
    assert 'class ContactModelForm' in artifacts['forms.py']
    
    # Verify HTML content
    assert 'ContactModel' in artifacts['list_create.html']


def test_generate_with_special_fields():
    """Test generation with various field types."""
    fields = [
        {'name': 'date_of_birth', 'type': 'DateField'},
        {'name': 'is_active', 'type': 'BooleanField'},
        {'name': 'description', 'type': 'TextField'},
        {'name': 'price', 'type': 'FloatField'},
    ]
    
    code = generate_model_code(fields, 'ProductModel')
    
    # Verify all field types are present
    assert 'DateField' in code
    assert 'BooleanField' in code
    assert 'TextField' in code
    assert 'FloatField' in code
    
    # Verify valid Python syntax
    try:
        ast.parse(code)
    except SyntaxError as e:
        pytest.fail(f"Generated code is not valid Python: {e}")


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])

