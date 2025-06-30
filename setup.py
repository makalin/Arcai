#!/usr/bin/env python3
"""
Arcai Setup Script
Installation and configuration script for the Arcai archaeological site detection system.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_banner():
    """Print the Arcai banner."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                    🔍 Arcai Setup 🔍                        ║
    ║                                                              ║
    ║         AI-Powered Archaeological Site Detection             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def create_directories():
    """Create necessary directories."""
    directories = [
        'data/positive',
        'data/negative',
        'data/sample_images',
        'models',
        'outputs',
        'uploads',
        'templates',
        'static/css',
        'static/js'
    ]
    
    print("📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")

def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        # Install from requirements.txt
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_sample_data():
    """Create sample data for testing."""
    print("📊 Creating sample data...")
    
    # Create a sample image (placeholder)
    sample_image_path = Path('data/sample_images/sample_satellite.tif')
    if not sample_image_path.exists():
        try:
            import numpy as np
            from PIL import Image
            
            # Create a sample satellite image
            sample_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            Image.fromarray(sample_image).save(sample_image_path)
            print(f"   ✅ Created sample image: {sample_image_path}")
        except Exception as e:
            print(f"   ⚠️  Could not create sample image: {e}")

def test_installation():
    """Test the installation."""
    print("🧪 Testing installation...")
    
    try:
        # Test imports
        import tensorflow as tf
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        import folium
        from PIL import Image
        
        print("   ✅ All required packages imported successfully")
        
        # Test model creation
        sys.path.append('src')
        from model import ArcaiNet
        
        model = ArcaiNet()
        print("   ✅ ArcaiNet model created successfully")
        
        return True
    except Exception as e:
        print(f"   ❌ Installation test failed: {e}")
        return False

def create_config_files():
    """Create configuration files."""
    print("⚙️  Creating configuration files...")
    
    # Create .env file
    env_content = """# Arcai Configuration
FLASK_ENV=development
FLASK_DEBUG=True
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    print("   ✅ Created .env file")

def create_scripts():
    """Create executable scripts."""
    print("📝 Creating executable scripts...")
    
    # Make Python files executable
    scripts = ['src/detect.py', 'src/train.py', 'web_interface.py']
    
    for script in scripts:
        if os.path.exists(script):
            os.chmod(script, 0o755)
            print(f"   ✅ Made {script} executable")

def print_next_steps():
    """Print next steps for the user."""
    print("""
    🎉 Installation Complete!
    
    Next Steps:
    ===========
    
    1️⃣  Add Training Data:
       - Place satellite images with archaeological sites in: data/positive/
       - Place images without sites in: data/negative/
    
    2️⃣  Train the Model:
       python src/train.py --data-dir data/ --epochs 50
    
    3️⃣  Run Detection:
       python src/detect.py --input data/sample_images/sample_satellite.tif
    
    4️⃣  Start Web Interface:
       python web_interface.py
    
    5️⃣  Open in Browser:
       http://localhost:5000
    
    Documentation:
    =============
    - README.md: Main documentation
    - data/README.md: Data format and organization
    - models/README.md: Model information
    - outputs/README.md: Output file descriptions
    
    Support:
    ========
    - GitHub Issues: Report bugs and request features
    - Documentation: Check README files for detailed guides
    
    Happy Archaeological Discovery! 🔍🏛️
    """)

def main():
    """Main setup function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Create sample data
    create_sample_data()
    
    # Test installation
    if not test_installation():
        print("❌ Setup failed during testing")
        sys.exit(1)
    
    # Create configuration files
    create_config_files()
    
    # Create executable scripts
    create_scripts()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main() 