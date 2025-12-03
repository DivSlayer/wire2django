"""
Flask web application for wireframe to Django code generator.

Routes:
    GET  /          - Upload form
    POST /upload    - Handle image upload and detection
    GET  /review    - Review detected fields with overlay
    POST /generate  - Generate Django code and return ZIP
"""
import os
import logging
import zipfile
import io
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, flash, session
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from pipeline import detect_fields
from generator import generate
from utils import sanitize_identifier, ensure_unique_field_names

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For flash messages and session

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5MB

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model_name_from_filename(filename: str) -> str:
    """
    Derive a friendly model name from the uploaded filename.
    
    Args:
        filename: Original filename (e.g., "user_form.jpg")
        
    Returns:
        Model name like "UserFormModel"
    """
    # Remove extension
    base_name = filename.rsplit('.', 1)[0]
    # Replace spaces and hyphens with underscores
    base_name = base_name.replace(' ', '_').replace('-', '_')
    # Capitalize each word and append Model
    parts = [p.capitalize() for p in base_name.split('_') if p]
    model_name = ''.join(parts) + "Model"
    return model_name if model_name else "AutoModel"


@app.route('/')
def index():
    """Display upload form."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and run detection pipeline."""
    if 'file' not in request.files:
        flash('No file uploaded. Please select an image file.')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected. Please choose an image file.')
        return redirect(url_for('index'))
    
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload a PNG, JPG, JPEG, GIF, or BMP image.')
        return redirect(url_for('index'))
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Saved uploaded file: {filepath}")
        
        # Run detection pipeline
        fields = detect_fields(filepath)
        
        if not fields:
            flash('No UI elements detected. Try uploading a clearer image with visible rectangles and labels.')
            return redirect(url_for('index'))
        
        # Store in session for review and generation
        session['uploaded_filename'] = filename
        session['fields'] = fields
        session['image_path'] = filepath
        
        logger.info(f"Detected {len(fields)} fields, redirecting to review")
        return redirect(url_for('review'))
        
    except Exception as e:
        logger.error(f"Error processing upload: {e}", exc_info=True)
        flash(f'Error processing image: {str(e)}. Please try again with a clearer image.')
        return redirect(url_for('index'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded images."""
    return send_file(os.path.join(UPLOAD_FOLDER, filename))


@app.route('/review', methods=['GET'])
def review():
    """Display review page with image overlay and editable fields."""
    if 'fields' not in session or 'image_path' not in session:
        flash('No image uploaded. Please upload an image first.')
        return redirect(url_for('index'))
    
    fields = session.get('fields', [])
    image_path = session.get('image_path')
    filename = session.get('uploaded_filename')
    
    return render_template('review.html', fields=fields, image_path=image_path, filename=filename)


@app.route('/generate', methods=['POST'])
def generate_code():
    """Generate Django code from reviewed fields and return as ZIP."""
    if 'fields' not in session:
        flash('No fields data found. Please upload an image first.')
        return redirect(url_for('index'))
    
    try:
        # Get field data from form
        field_ids = request.form.getlist('field_id')
        field_names = request.form.getlist('field_name')
        field_types = request.form.getlist('field_type')
        
        # Reconstruct fields list from form data
        reviewed_fields = []
        for i, field_id in enumerate(field_ids):
            if i < len(field_names) and i < len(field_types):
                field_name = sanitize_identifier(field_names[i].strip())
                field_type = field_types[i].strip()
                
                # Skip empty fields
                if not field_name:
                    continue
                
                reviewed_fields.append({
                    'id': int(field_id),
                    'name': field_name,
                    'type': field_type,
                    'suggested_name': field_name,
                    'suggested_type': field_type
                })
        
        if not reviewed_fields:
            flash('No valid fields found. Please ensure all fields have names.')
            return redirect(url_for('review'))
        
        # Ensure unique field names
        reviewed_fields = ensure_unique_field_names(reviewed_fields)
        
        # Get model name from filename or use default
        filename = session.get('uploaded_filename', 'wireframe.jpg')
        model_name = get_model_name_from_filename(filename)
        
        logger.info(f"Generating code for {len(reviewed_fields)} fields with model name: {model_name}")
        
        # Generate all Django code artifacts
        code_files = generate(reviewed_fields, model_name)
        
        # Create ZIP in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for filename, content in code_files.items():
                zip_file.writestr(filename, content)
        
        zip_buffer.seek(0)
        
        # Return ZIP file for download
        zip_filename = f"{model_name.lower()}_generated.zip"
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )
        
    except Exception as e:
        logger.error(f"Error generating code: {e}", exc_info=True)
        flash(f'Error generating code: {str(e)}. Please try again.')
        return redirect(url_for('review'))


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """Handle file size limit exceeded."""
    flash('File too large. Maximum size is 5MB.')
    return redirect(url_for('index'))


@app.errorhandler(413)
def request_entity_too_large(e):
    """Handle 413 error."""
    flash('File too large. Maximum size is 5MB.')
    return redirect(url_for('index'))


if __name__ == '__main__':
    logger.info("Starting Wire2Django application...")
    app.run(debug=True, host='127.0.0.1', port=5000)

