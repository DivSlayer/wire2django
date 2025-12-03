# Quick Start Guide

## Prerequisites Check

1. **Python 3.11+**
   ```bash
   python3 --version
   ```

2. **Tesseract OCR**
   ```bash
   tesseract --version
   ```

## Installation Steps

```bash
# Navigate to project directory
cd wire2django

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

### Option 1: Direct Python
```bash
python app.py
```

### Option 2: Using the script
```bash
./run.sh
```

Then open: http://127.0.0.1:5000

## Testing with Sample Images

1. Start the application
2. Go to http://127.0.0.1:5000
3. Upload one of the sample images from `sample_images/`:
   - `sketch1_login.jpg` - Simple login form
   - `sketch2_registration.jpg` - Registration form
   - `sketch3_contact.jpg` - Contact form
4. Review detected fields
5. Edit field names/types if needed
6. Click "Generate Django Code & Download ZIP"

## Running Tests

```bash
pytest tests/ -v
```

## Example Workflow

1. **Upload**: Upload `sample_images/sketch1_login.jpg`
2. **Review**: See detected fields "Email" and "Password"
3. **Edit**: Verify field names and types (EmailField, CharField)
4. **Generate**: Download ZIP containing:
   - `models.py` - Django model with email and password fields
   - `forms.py` - Django ModelForm
   - `list_create.html` - HTML template

## Troubleshooting

### "TesseractNotFoundError"
- Install Tesseract: `brew install tesseract` (macOS) or `sudo apt-get install tesseract-ocr` (Ubuntu)
- Verify: `tesseract --version`

### No fields detected
- Use clearer image with high contrast
- Draw rectangles clearly with dark lines
- Write labels in English clearly

### Import errors
- Make sure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

