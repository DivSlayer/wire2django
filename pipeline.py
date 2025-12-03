"""
Image processing pipeline for detecting UI elements in wireframe images.

Uses OpenCV for contour detection and Tesseract OCR for text recognition.
"""
import cv2
import numpy as np
import pytesseract
from PIL import Image
import logging
from typing import List, Dict, Tuple
import re

from utils import slugify_name, sanitize_identifier

logger = logging.getLogger(__name__)

# Configuration constants
MIN_CONTOUR_AREA = 1000  # Minimum area in pixels for a valid UI element
MAX_CONTOUR_AREA = 500000  # Maximum area to avoid detecting entire page
OCR_PSM = 6  # Page segmentation mode: assume single uniform block
ROI_PADDING = 10  # Pixels to add around bounding box for OCR


def detect_fields(image_path: str) -> List[Dict]:
    """
    Detect rectangular UI elements and extract text from a wireframe image.
    
    Process:
    1. Load and preprocess image (grayscale, threshold)
    2. Find contours (rectangular shapes)
    3. Filter by size and shape
    4. Extract text using OCR for each region
    5. Infer field type from text
    6. Sanitize field names
    
    Args:
        image_path: Path to the wireframe image file
        
    Returns:
        List of dicts, each containing:
            - id: unique integer identifier
            - text: raw OCR text
            - bbox: tuple (x, y, width, height)
            - suggested_name: sanitized field name
            - suggested_type: Django field type suggestion
    """
    logger.info(f"Processing image: {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to load image: {image_path}")
        return []
    
    original_height, original_width = img.shape[:2]
    logger.debug(f"Image size: {original_width}x{original_height}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive threshold to get binary image
    # This helps with hand-drawn sketches and varying lighting
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Optional: Apply morphological operations to clean up noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.debug(f"Found {len(contours)} contours")
    
    fields = []
    field_id = 1
    
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if it's roughly rectangular (4 corners, or close enough)
        # For hand-drawn sketches, be lenient
        if len(approx) >= 4:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Additional validation: reasonable aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                continue  # Skip very thin or very tall rectangles
            
            # Extract ROI for OCR (with padding)
            x_start = max(0, x - ROI_PADDING)
            y_start = max(0, y - ROI_PADDING)
            x_end = min(original_width, x + w + ROI_PADDING)
            y_end = min(original_height, y + h + ROI_PADDING)
            
            roi = gray[y_start:y_end, x_start:x_end]
            
            # Skip if ROI is too small
            if roi.size < 100:
                continue
            
            # Perform OCR on the ROI
            try:
                # Use Tesseract with specific PSM mode for single text block
                ocr_text = pytesseract.image_to_string(
                    roi, 
                    lang='eng', 
                    config=f'--psm {OCR_PSM}'
                ).strip()
                
                logger.debug(f"OCR text for bbox ({x}, {y}, {w}, {h}): '{ocr_text}'")
                
                # If OCR returned text, create field entry
                if ocr_text:
                    suggested_name = sanitize_identifier(
                        slugify_name(ocr_text), 
                        fallback_prefix="field"
                    )
                    suggested_type = infer_field_type(ocr_text)
                    
                    fields.append({
                        'id': field_id,
                        'text': ocr_text,
                        'bbox': (x, y, w, h),
                        'suggested_name': suggested_name,
                        'suggested_type': suggested_type
                    })
                    field_id += 1
                    
            except Exception as e:
                logger.warning(f"OCR failed for bbox ({x}, {y}, {w}, {h}): {e}")
                continue
    
    logger.info(f"Detected {len(fields)} fields")
    return fields


def infer_field_type(text: str) -> str:
    """
    Infer Django field type from OCR text using heuristics.
    
    Heuristic rules:
    - Contains "email" or "@" -> EmailField
    - Contains "password" -> CharField (with password widget)
    - Contains "date", "dob", "day" -> DateField
    - Contains "age", "number", "qty", or numeric-only -> IntegerField
    - Long text or paragraph-like -> TextField
    - Default -> CharField
    
    Args:
        text: Raw OCR text from the wireframe
        
    Returns:
        Django field type string (e.g., "EmailField", "CharField")
    """
    if not text:
        return "CharField"
    
    text_lower = text.lower().strip()
    
    # Email detection
    if '@' in text or 'email' in text_lower or 'e-mail' in text_lower:
        return "EmailField"
    
    # Password detection
    if 'password' in text_lower or 'pass' in text_lower:
        return "CharField"  # Will use password widget in forms
    
    # Date detection
    date_keywords = ['date', 'dob', 'birth', 'day', 'time', 'when']
    if any(keyword in text_lower for keyword in date_keywords):
        return "DateField"
    
    # Number/integer detection
    numeric_keywords = ['age', 'number', 'qty', 'quantity', 'count', 'num', 'int', 'integer']
    if any(keyword in text_lower for keyword in numeric_keywords):
        return "IntegerField"
    
    # Check if text is numeric-only (could be integer)
    if re.match(r'^\d+$', text.strip()):
        return "IntegerField"
    
    # Check for float indicators
    if '.' in text and re.match(r'^[\d.]+$', text.strip()):
        return "FloatField"
    
    # Boolean detection (common patterns)
    bool_keywords = ['yes', 'no', 'true', 'false', 'checkbox', 'check', 'agree']
    if any(keyword in text_lower for keyword in bool_keywords):
        return "BooleanField"
    
    # Long text detection (TextField)
    if len(text) > 50 or '\n' in text or text_lower in ['description', 'notes', 'comment', 'message', 'text']:
        return "TextField"
    
    # Default to CharField
    return "CharField"


# Stub mode for testing (set USE_STUB=True to bypass OpenCV/Tesseract)
USE_STUB = False

def detect_fields_stub(image_path: str) -> List[Dict]:
    """
    Stub implementation for testing without OpenCV/Tesseract dependencies.
    Returns mock fields based on filename or hardcoded data.
    """
    logger.info(f"Using stub detection for: {image_path}")
    
    # Return some mock fields for testing
    return [
        {
            'id': 1,
            'text': 'Email',
            'bbox': (50, 50, 200, 30),
            'suggested_name': 'email',
            'suggested_type': 'EmailField'
        },
        {
            'id': 2,
            'text': 'Password',
            'bbox': (50, 100, 200, 30),
            'suggested_name': 'password',
            'suggested_type': 'CharField'
        },
        {
            'id': 3,
            'text': 'Full Name',
            'bbox': (50, 150, 200, 30),
            'suggested_name': 'full_name',
            'suggested_type': 'CharField'
        }
    ]


# Use stub if USE_STUB is True
if USE_STUB:
    detect_fields = detect_fields_stub

