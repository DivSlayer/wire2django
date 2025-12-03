# Wire2Django

A Flask-based MVP tool that converts hand-drawn wireframe images into Django code artifacts (models, forms, and templates). Upload a wireframe image, review detected fields, and download a ZIP containing ready-to-use Django code.

## Features

- **Image Upload**: Upload wireframe images (PNG, JPG, JPEG, GIF, BMP)
- **Automatic Detection**: Uses OpenCV for rectangle detection and Tesseract OCR for text recognition
- **Field Type Inference**: Automatically suggests Django field types based on OCR text heuristics
- **Interactive Review**: Edit field names and types before generation
- **Code Generation**: Generates `models.py`, `forms.py`, and HTML templates using Jinja2
- **ZIP Download**: Packages all generated files into a downloadable ZIP

## Prerequisites

### System Requirements

- **Python 3.11+**
- **Tesseract OCR** (must be installed separately)

### Installing Tesseract OCR

#### macOS
```bash
brew install tesseract
```

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

#### Windows
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install and add to PATH
3. Note the installation path (usually `C:\Program Files\Tesseract-OCR`)

#### Verify Installation
```bash
tesseract --version
```

## Installation

1. **Clone or navigate to the project directory:**
```bash
cd wire2django
```

2. **Create a virtual environment (recommended):**
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Running the Application

1. **Start the Flask server:**
```bash
python app.py
```

2. **Open your browser:**
```
http://127.0.0.1:5000
```

3. **Upload a wireframe image:**
   - Click "Choose File" and select an image
   - Click "Upload & Detect"
   - Review detected fields on the review page
   - Edit field names and types as needed
   - Click "Generate Django Code & Download ZIP"

## Sample Images

The project includes three sample wireframe images in `sample_images/`:

- **sketch1_login.jpg**: Simple login form with Email and Password fields
- **sketch2_registration.jpg**: User registration form with multiple fields
- **sketch3_contact.jpg**: Contact form with Name, Email, Phone, and Message fields

You can upload these to test the application immediately.

## Tips for Best Results

### Drawing Your Wireframe

- Use a **dark pen** on **white paper** or background
- Draw **clear rectangular boxes** for form fields
- Write **labels clearly** above or inside each box
- Keep labels in **English** only
- Ensure **good lighting** and **high contrast** in photos
- Avoid excessive noise or clutter
- Use consistent spacing between elements

### Field Detection

The system uses heuristics to infer field types:

- **EmailField**: Contains "email", "@", or "e-mail"
- **CharField**: Default for short text (or "password" for password fields)
- **TextField**: Long text or keywords like "description", "notes"
- **IntegerField**: Keywords like "age", "number", "qty", or numeric-only text
- **FloatField**: Numbers with decimal points
- **DateField**: Keywords like "date", "dob", "birth", "day"
- **BooleanField**: Keywords like "yes", "no", "checkbox", "agree"

## Generated Code Structure

The generated ZIP contains:

```
generated_code.zip
├── models.py          # Django model class
├── forms.py           # Django ModelForm
└── list_create.html   # HTML template for list/create view
```

### Example Generated models.py

```python
from django.db import models

class LoginFormModel(models.Model):
    email = models.EmailField(max_length=254, blank=True, null=True)
    password = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        verbose_name = "LoginFormModel"
        verbose_name_plural = "LoginFormModels"

    def __str__(self):
        first_field = getattr(self, 'email', None)
        if first_field:
            return str(first_field)
        return str(self.id)
```

## Using Generated Code

1. **Extract the ZIP** to your Django project
2. **Add models.py** to your app's `models.py` or import it
3. **Add forms.py** to your app's `forms.py` or import it
4. **Add the HTML template** to your `templates/` directory
5. **Run migrations:**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```
6. **Create views** to use the generated forms and templates

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Tests cover:
- Field type inference heuristics
- Field name sanitization
- Django code generation (models, forms, templates)
- Code syntax validation

## Project Structure

```
wire2django/
├── app.py                    # Flask application with routes
├── pipeline.py               # OpenCV + Tesseract detection logic
├── generator.py              # Jinja2 code generation
├── utils.py                  # Helper functions (sanitization, etc.)
├── requirements.txt          # Python dependencies
├── create_sample_images.py   # Script to generate sample wireframes
├── templates/                # Flask HTML templates
│   ├── index.html
│   └── review.html
├── templates_code/           # Jinja2 templates for Django code
│   ├── model_template.jinja2
│   ├── form_template.jinja2
│   └── html_template.jinja2
├── static/                   # CSS and static assets
│   └── style.css
├── tests/                    # Unit tests
│   ├── test_pipeline.py
│   └── test_generator.py
├── sample_images/            # Example wireframe images
│   ├── sketch1_login.jpg
│   ├── sketch2_registration.jpg
│   └── sketch3_contact.jpg
├── uploads/                  # Temporary upload storage (created at runtime)
└── README.md                 # This file
```

## Troubleshooting

### No Fields Detected

- **Check image quality**: Ensure high contrast and clear labels
- **Verify rectangle visibility**: Make sure boxes are clearly drawn
- **Increase image size**: Larger images often work better
- **Check OCR language**: Currently supports English only

### OCR Returns Garbled Text

- Edit field names manually on the review page before generating
- Use clearer handwriting or printed text
- Increase image resolution
- Improve lighting in your photo

### Tesseract Not Found

If you get an error like `TesseractNotFoundError`:

1. Verify Tesseract is installed: `tesseract --version`
2. On some systems, you may need to set the path explicitly:
   ```python
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Linux
   # or
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
   ```

### Detection Accuracy Issues

- Draw rectangles with clear boundaries (thick lines help)
- Place labels consistently (above or inside boxes)
- Avoid overlapping elements
- Use simple, standard form layouts

## Limitations (MVP)

- **English only**: OCR and UI support English text only
- **Simple rectangles**: Detects rectangular UI elements only (no circles, complex shapes)
- **No validation**: Generated code doesn't include field validation beyond basic types
- **Static templates**: Generated HTML templates are basic placeholders
- **No database operations**: Generated code is meant to be integrated into your Django app

## Next Steps / Future Enhancements

### Suggested Improvements

1. **Better Detection Models**:
   - Integrate YOLO object detection for more accurate element recognition
   - Use pix2struct or similar models for better OCR and layout understanding
   - Support for more complex UI elements (buttons, dropdowns, checkboxes)

2. **Enhanced Field Types**:
   - Support for ForeignKey relationships
   - File upload fields
   - JSON fields
   - Custom validators

3. **Code Generation**:
   - Generate complete Django views (CreateView, ListView, etc.)
   - Generate URL configurations
   - Generate migrations automatically
   - Support for Django REST Framework serializers

4. **UI Improvements**:
   - Drag-and-drop field reordering
   - Real-time preview of generated code
   - Export to different formats (JSON, YAML)

5. **Multi-language Support**:
   - Support for multiple OCR languages
   - Internationalization of the web UI

### Adding YOLO or pix2struct

To improve detection accuracy:

1. **Train or use a pre-trained YOLO model** for UI element detection
2. **Replace or enhance** the `detect_fields()` function in `pipeline.py`
3. **Integrate pix2struct** for better OCR and layout understanding:
   ```python
   from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
   # Process image and extract structured information
   ```

## Development

### Running in Development Mode

```bash
export FLASK_ENV=development
python app.py
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for public functions
- Include docstrings for all modules and functions
- Keep functions focused and single-purpose

## License

This is an MVP project for demonstration purposes.

## Contributing

This is an MVP project. Suggestions and improvements are welcome!

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the test files for usage examples
3. Check that all dependencies are correctly installed

---

**Note**: This is an MVP (Minimum Viable Product). The generated code should be reviewed and customized before use in production applications.

