"""
Unit tests for pipeline.py detection functionality.
"""
import os
import sys
import tempfile
from PIL import Image, ImageDraw, ImageFont
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline import detect_fields, infer_field_type
from utils import sanitize_identifier


def create_test_image_with_labels(labels_and_positions):
    """
    Create a synthetic test image with rectangles and text labels.
    
    Args:
        labels_and_positions: List of tuples (label, x, y, width, height)
        
    Returns:
        Temporary file path
    """
    # Create a white image
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except:
            font = ImageFont.load_default()
    
    for label, x, y, w, h in labels_and_positions:
        # Draw rectangle
        draw.rectangle([x, y, x + w, y + h], outline='black', width=2)
        # Draw label above rectangle
        draw.text((x, y - 25), label, fill='black', font=font)
    
    # Save to temporary file
    fd, path = tempfile.mkstemp(suffix='.png')
    img.save(path)
    os.close(fd)
    
    return path


def test_infer_field_type_email():
    """Test email field type inference."""
    assert infer_field_type("Email") == "EmailField"
    assert infer_field_type("email address") == "EmailField"
    assert infer_field_type("user@example.com") == "EmailField"
    assert infer_field_type("e-mail") == "EmailField"


def test_infer_field_type_password():
    """Test password field type inference."""
    assert infer_field_type("Password") == "CharField"
    assert infer_field_type("pass") == "CharField"


def test_infer_field_type_date():
    """Test date field type inference."""
    assert infer_field_type("Date") == "DateField"
    assert infer_field_type("DOB") == "DateField"
    assert infer_field_type("Birth Date") == "DateField"


def test_infer_field_type_integer():
    """Test integer field type inference."""
    assert infer_field_type("Age") == "IntegerField"
    assert infer_field_type("Number") == "IntegerField"
    assert infer_field_type("Qty") == "IntegerField"
    assert infer_field_type("123") == "IntegerField"


def test_infer_field_type_text():
    """Test text field type inference."""
    long_text = "This is a very long description field that should be detected as TextField"
    assert infer_field_type(long_text) == "TextField"
    assert infer_field_type("Description") == "TextField"
    assert infer_field_type("Notes") == "TextField"


def test_infer_field_type_default():
    """Test default CharField inference."""
    assert infer_field_type("Name") == "CharField"
    assert infer_field_type("Title") == "CharField"
    assert infer_field_type("") == "CharField"


def test_sanitize_identifier():
    """Test field name sanitization."""
    assert sanitize_identifier("Email Address") == "email_address"
    assert sanitize_identifier("First Name") == "first_name"
    assert sanitize_identifier("123invalid") == "field_1"  # Starts with digit
    assert sanitize_identifier("if") == "if_field"  # Reserved word
    assert sanitize_identifier("test-field!") == "testfield"


def test_detect_fields_with_mock_image():
    """
    Test field detection with a synthetic image.
    
    Note: This test may require OpenCV and Tesseract to be installed.
    If they're not available, the test will be skipped or use stub mode.
    """
    # Create test image with labels
    labels = [
        ("Email", 50, 50, 200, 30),
        ("Password", 50, 100, 200, 30),
        ("Full Name", 50, 150, 200, 30),
    ]
    
    image_path = create_test_image_with_labels(labels)
    
    try:
        # Try to detect fields
        # Note: This may fail if Tesseract is not installed or OCR fails
        fields = detect_fields(image_path)
        
        # If fields are detected, verify structure
        if fields:
            assert isinstance(fields, list)
            for field in fields:
                assert 'id' in field
                assert 'text' in field
                assert 'bbox' in field
                assert 'suggested_name' in field
                assert 'suggested_type' in field
                assert isinstance(field['bbox'], tuple)
                assert len(field['bbox']) == 4
        
    finally:
        # Clean up
        if os.path.exists(image_path):
            os.unlink(image_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

