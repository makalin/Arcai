#!/usr/bin/env python3
"""
Arcai Web Interface
A simple Flask web application for archaeological site detection.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import uuid

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import ArcaiNet
from src.preprocessing import preprocess_image
from src.visualization import visualize_detections, create_interactive_map, plot_detection_confidence

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# Global model variable
model = None

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the ArcaiNet model."""
    global model
    try:
        # Try to load pre-trained model
        model_path = 'models/arcainet_v1.h5'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded from {model_path}")
        else:
            # Create new model if pre-trained doesn't exist
            model = ArcaiNet()
            print("✅ New ArcaiNet model created")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = ArcaiNet()
        print("✅ Fallback: New ArcaiNet model created")

def detect_sites(image_path, confidence_threshold=0.5):
    """Detect archaeological sites in image."""
    global model
    
    if model is None:
        load_model()
    
    try:
        # Preprocess image
        image = preprocess_image(image_path)
        
        # Make prediction
        prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)
        confidence = float(prediction[0][0])
        
        # Create detection results
        detections = []
        if confidence > confidence_threshold:
            detections.append({
                'box_id': 1,
                'confidence': confidence,
                'bbox': [0, 0, image.shape[1], image.shape[0]]
            })
        
        results = {
            'image_path': image_path,
            'detections': detections,
            'total_sites': len(detections),
            'timestamp': datetime.now().isoformat()
        }
        
        return results, None
        
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and detection."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{timestamp}_{unique_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get confidence threshold
        confidence_threshold = float(request.form.get('confidence', 0.5))
        
        # Run detection
        results, error = detect_sites(filepath, confidence_threshold)
        
        if error:
            return jsonify({'error': f'Detection failed: {error}'}), 500
        
        # Generate outputs
        output_id = f"{timestamp}_{unique_id}"
        
        # Visualization
        viz_path = f"outputs/detection_result_{output_id}.png"
        visualize_detections(filepath, results, viz_path, confidence_threshold)
        
        # Interactive map
        map_path = f"outputs/interactive_map_{output_id}.html"
        create_interactive_map(filepath, results, map_path)
        
        # Confidence plot
        conf_path = f"outputs/confidence_plot_{output_id}.png"
        plot_detection_confidence(results, conf_path)
        
        # Save results JSON
        json_path = f"outputs/results_{output_id}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return jsonify({
            'success': True,
            'output_id': output_id,
            'results': results,
            'files': {
                'visualization': viz_path,
                'interactive_map': map_path,
                'confidence_plot': conf_path,
                'json_results': json_path
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/results/<output_id>')
def view_results(output_id):
    """View detection results."""
    # Load results
    json_path = f"outputs/results_{output_id}.json"
    if not os.path.exists(json_path):
        return "Results not found", 404
    
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    return render_template('results.html', results=results, output_id=output_id)

@app.route('/download/<output_id>/<file_type>')
def download_file(output_id, file_type):
    """Download output files."""
    file_mapping = {
        'visualization': f"outputs/detection_result_{output_id}.png",
        'interactive_map': f"outputs/interactive_map_{output_id}.html",
        'confidence_plot': f"outputs/confidence_plot_{output_id}.png",
        'json_results': f"outputs/results_{output_id}.json"
    }
    
    if file_type not in file_mapping:
        return "File type not found", 404
    
    file_path = file_mapping[file_type]
    if not os.path.exists(file_path):
        return "File not found", 404
    
    return send_file(file_path, as_attachment=True)

@app.route('/api/status')
def api_status():
    """API status endpoint."""
    global model
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000) 