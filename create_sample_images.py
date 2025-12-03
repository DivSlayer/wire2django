"""
Script to create sample wireframe images for testing and demonstration.

Run this script to generate sample wireframe images in the sample_images/ directory.
"""
import os
from PIL import Image, ImageDraw, ImageFont

def create_sample_image_1():
    """Create a simple login form wireframe."""
    img = Image.new('RGB', (600, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a readable font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except:
            font = ImageFont.load_default()
    
    # Draw form elements
    # Email field
    draw.rectangle([50, 100, 350, 140], outline='black', width=2)
    draw.text((50, 70), "Email", fill='black', font=font)
    
    # Password field
    draw.rectangle([50, 200, 350, 240], outline='black', width=2)
    draw.text((50, 170), "Password", fill='black', font=font)
    
    # Title
    draw.text((50, 20), "Login Form", fill='black', font=font)
    
    return img


def create_sample_image_2():
    """Create a user registration form wireframe."""
    img = Image.new('RGB', (700, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        except:
            font = ImageFont.load_default()
    
    # Title
    draw.text((50, 20), "User Registration", fill='black', font=font)
    
    # First Name
    draw.rectangle([50, 80, 300, 120], outline='black', width=2)
    draw.text((50, 50), "First Name", fill='black', font=font)
    
    # Last Name
    draw.rectangle([350, 80, 600, 120], outline='black', width=2)
    draw.text((350, 50), "Last Name", fill='black', font=font)
    
    # Email
    draw.rectangle([50, 160, 600, 200], outline='black', width=2)
    draw.text((50, 130), "Email Address", fill='black', font=font)
    
    # Date of Birth
    draw.rectangle([50, 240, 250, 280], outline='black', width=2)
    draw.text((50, 210), "Date of Birth", fill='black', font=font)
    
    # Age
    draw.rectangle([300, 240, 400, 280], outline='black', width=2)
    draw.text((300, 210), "Age", fill='black', font=font)
    
    # Description
    draw.rectangle([50, 320, 600, 420], outline='black', width=2)
    draw.text((50, 290), "Description", fill='black', font=font)
    
    return img


def create_sample_image_3():
    """Create a contact form wireframe."""
    img = Image.new('RGB', (650, 500), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        except:
            font = ImageFont.load_default()
    
    # Title
    draw.text((50, 20), "Contact Us", fill='black', font=font)
    
    # Name
    draw.rectangle([50, 80, 550, 120], outline='black', width=2)
    draw.text((50, 50), "Full Name", fill='black', font=font)
    
    # Email
    draw.rectangle([50, 160, 550, 200], outline='black', width=2)
    draw.text((50, 130), "Email", fill='black', font=font)
    
    # Phone Number
    draw.rectangle([50, 240, 550, 280], outline='black', width=2)
    draw.text((50, 210), "Phone Number", fill='black', font=font)
    
    # Message
    draw.rectangle([50, 320, 550, 420], outline='black', width=2)
    draw.text((50, 290), "Message", fill='black', font=font)
    
    return img


def main():
    """Generate all sample images."""
    sample_dir = 'sample_images'
    os.makedirs(sample_dir, exist_ok=True)
    
    print("Creating sample wireframe images...")
    
    # Create sample 1: Login form
    img1 = create_sample_image_1()
    img1.save(os.path.join(sample_dir, 'sketch1_login.jpg'), 'JPEG', quality=95)
    print(f"Created {sample_dir}/sketch1_login.jpg")
    
    # Create sample 2: Registration form
    img2 = create_sample_image_2()
    img2.save(os.path.join(sample_dir, 'sketch2_registration.jpg'), 'JPEG', quality=95)
    print(f"Created {sample_dir}/sketch2_registration.jpg")
    
    # Create sample 3: Contact form
    img3 = create_sample_image_3()
    img3.save(os.path.join(sample_dir, 'sketch3_contact.jpg'), 'JPEG', quality=95)
    print(f"Created {sample_dir}/sketch3_contact.jpg")
    
    print("\nSample images created successfully!")
    print("You can now upload these images to test the wireframe detection.")


if __name__ == '__main__':
    main()

