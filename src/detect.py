#!/usr/bin/env python3
"""
Arcai - Archaeological Site Detection
Main detection script for analyzing satellite imagery.
"""

import argparse
import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ArcaiNet
from src.preprocessing import preprocess_image
from src.visualization import visualize_detections


def load_model(model_path):
    """Load the pre-trained ArcaiNet model."""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Using default ArcaiNet model...")
        return ArcaiNet()


def detect_sites(image_path, model, confidence_threshold=0.5):
    """
    Detect potential archaeological sites in satellite imagery.
    
    Args:
        image_path (str): Path to satellite image
        model: Pre-trained detection model
        confidence_threshold (float): Minimum confidence for detections
    
    Returns:
        dict: Detection results with bounding boxes and confidence scores
    """
    print(f"🔍 Analyzing image: {image_path}")
    
    # Load and preprocess image
    image = preprocess_image(image_path)
    
    # Run detection
    predictions = model.predict(np.expand_dims(image, axis=0))
    
    # Process predictions
    detections = []
    for i, confidence in enumerate(predictions[0]):
        if confidence > confidence_threshold:
            detections.append({
                'box_id': i,
                'confidence': float(confidence),
                'bbox': [0, 0, image.shape[1], image.shape[0]]  # Simplified bbox
            })
    
    print(f"🎯 Found {len(detections)} potential archaeological sites")
    return {
        'image_path': image_path,
        'detections': detections,
        'total_sites': len(detections)
    }


def main():
    parser = argparse.ArgumentParser(description='Arcai - Archaeological Site Detection')
    parser.add_argument('--input', '-i', required=True, help='Input satellite image path')
    parser.add_argument('--model', '-m', default='models/arcainet_v1.h5', help='Model path')
    parser.add_argument('--output', '-o', default='outputs/detection_result.png', help='Output visualization path')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--save-results', action='store_true', help='Save detection results to JSON')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return 1
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load model
    model = load_model(args.model)
    
    # Run detection
    results = detect_sites(args.input, model, args.confidence)
    
    # Visualize results
    print(f"📊 Creating visualization: {args.output}")
    visualize_detections(args.input, results, args.output)
    
    # Save results if requested
    if args.save_results:
        import json
        output_json = args.output.replace('.png', '.json')
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {output_json}")
    
    print("✅ Detection completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 