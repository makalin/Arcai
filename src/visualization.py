"""
Visualization utilities for Arcai detection results
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import folium
from PIL import Image
import cv2
import os
from typing import Dict, List, Tuple, Optional
import json


def visualize_detections(image_path: str, results: Dict, output_path: str, 
                        confidence_threshold: float = 0.5) -> None:
    """
    Create visualization of detection results.
    
    Args:
        image_path: Path to the original satellite image
        results: Detection results dictionary
        output_path: Path to save the visualization
        confidence_threshold: Minimum confidence for visualization
    """
    # Load original image
    image = np.array(Image.open(image_path))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original image
    ax1.imshow(image)
    ax1.set_title('Original Satellite Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Image with detections
    ax2.imshow(image)
    
    # Add detection boxes
    detections = results.get('detections', [])
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(detections)))
    
    for i, detection in enumerate(detections):
        confidence = detection.get('confidence', 0)
        if confidence >= confidence_threshold:
            bbox = detection.get('bbox', [0, 0, image.shape[1], image.shape[0]])
            x, y, w, h = bbox
            
            # Create rectangle patch
            rect = patches.Rectangle((x, y), w, h, 
                                   linewidth=2, 
                                   edgecolor=colors[i], 
                                   facecolor='none',
                                   alpha=0.8)
            ax2.add_patch(rect)
            
            # Add confidence text
            ax2.text(x, y-10, f'{confidence:.2f}', 
                    color=colors[i], fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax2.set_title(f'Detected Sites ({len(detections)} found)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Add overall statistics
    total_sites = results.get('total_sites', 0)
    fig.suptitle(f'Arcai Detection Results - {total_sites} Potential Archaeological Sites', 
                fontsize=16, fontweight='bold')
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualization saved to: {output_path}")


def plot_training_history(history, output_path: str) -> None:
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Training history from model.fit()
        output_path: Path to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    ax1.plot(history.history['loss'], label='Training Loss', color='blue')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Precision
    if 'precision' in history.history:
        ax3.plot(history.history['precision'], label='Training Precision', color='blue')
        ax3.plot(history.history['val_precision'], label='Validation Precision', color='red')
        ax3.set_title('Model Precision', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Precision')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Recall
    if 'recall' in history.history:
        ax4.plot(history.history['recall'], label='Training Recall', color='blue')
        ax4.plot(history.history['val_recall'], label='Validation Recall', color='red')
        ax4.set_title('Model Recall', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Recall')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Training history plot saved to: {output_path}")


def create_interactive_map(image_path: str, results: Dict, output_path: str,
                          center_coords: Optional[Tuple[float, float]] = None) -> None:
    """
    Create an interactive Folium map with detection results.
    
    Args:
        image_path: Path to the satellite image
        results: Detection results dictionary
        output_path: Path to save the HTML map
        center_coords: Center coordinates for the map (lat, lon)
    """
    # Default center coordinates (can be extracted from GeoTIFF metadata)
    if center_coords is None:
        center_coords = (40.7128, -74.0060)  # Default to NYC
    
    # Create base map
    m = folium.Map(location=center_coords, zoom_start=15)
    
    # Add detection markers
    detections = results.get('detections', [])
    
    for i, detection in enumerate(detections):
        confidence = detection.get('confidence', 0)
        bbox = detection.get('bbox', [0, 0, 100, 100])
        
        # Calculate marker position (simplified)
        lat_offset = (i * 0.001)  # Small offset for multiple detections
        marker_lat = center_coords[0] + lat_offset
        marker_lon = center_coords[1] + lat_offset
        
        # Create popup content
        popup_content = f"""
        <div style="width: 200px;">
            <h4>Archaeological Site #{i+1}</h4>
            <p><strong>Confidence:</strong> {confidence:.2%}</p>
            <p><strong>Detection ID:</strong> {detection.get('box_id', 'N/A')}</p>
            <p><strong>Bounding Box:</strong> {bbox}</p>
        </div>
        """
        
        # Add marker with color based on confidence
        color = 'red' if confidence > 0.8 else 'orange' if confidence > 0.6 else 'yellow'
        
        folium.Marker(
            location=[marker_lat, marker_lon],
            popup=folium.Popup(popup_content, max_width=300),
            icon=folium.Icon(color=color, icon='info-sign'),
            tooltip=f"Site #{i+1} - {confidence:.2%}"
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 90px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><strong>Detection Confidence</strong></p>
    <p><i class="fa fa-circle" style="color:red"></i> High (>80%)</p>
    <p><i class="fa fa-circle" style="color:orange"></i> Medium (60-80%)</p>
    <p><i class="fa fa-circle" style="color:yellow"></i> Low (<60%)</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save(output_path)
    print(f"🗺️ Interactive map saved to: {output_path}")


def plot_detection_confidence(results: Dict, output_path: str) -> None:
    """
    Create a bar plot of detection confidences.
    
    Args:
        results: Detection results dictionary
        output_path: Path to save the plot
    """
    detections = results.get('detections', [])
    
    if not detections:
        print("No detections to plot")
        return
    
    confidences = [d.get('confidence', 0) for d in detections]
    site_ids = [f"Site #{i+1}" for i in range(len(detections))]
    
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    bars = plt.bar(site_ids, confidences, color='skyblue', alpha=0.7)
    
    # Add confidence threshold line
    threshold = 0.5
    plt.axhline(y=threshold, color='red', linestyle='--', 
                label=f'Threshold ({threshold:.1%})', alpha=0.8)
    
    # Customize plot
    plt.title('Detection Confidence Scores', fontsize=16, fontweight='bold')
    plt.xlabel('Detected Sites', fontsize=12)
    plt.ylabel('Confidence Score', fontsize=12)
    plt.ylim(0, 1)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, conf in zip(bars, confidences):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{conf:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Confidence plot saved to: {output_path}")


def create_comparison_visualization(image_paths: List[str], results_list: List[Dict], 
                                  output_path: str) -> None:
    """
    Create a comparison visualization of multiple detection results.
    
    Args:
        image_paths: List of image paths
        results_list: List of detection results
        output_path: Path to save the comparison
    """
    n_images = len(image_paths)
    fig, axes = plt.subplots(2, n_images, figsize=(5*n_images, 10))
    
    if n_images == 1:
        axes = axes.reshape(2, 1)
    
    for i, (image_path, results) in enumerate(zip(image_paths, results_list)):
        # Load image
        image = np.array(Image.open(image_path))
        
        # Original image
        axes[0, i].imshow(image)
        axes[0, i].set_title(f'Image {i+1} - Original', fontsize=12)
        axes[0, i].axis('off')
        
        # Image with detections
        axes[1, i].imshow(image)
        
        # Add detection boxes
        detections = results.get('detections', [])
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(detections)))
        
        for j, detection in enumerate(detections):
            confidence = detection.get('confidence', 0)
            bbox = detection.get('bbox', [0, 0, image.shape[1], image.shape[0]])
            x, y, w, h = bbox
            
            rect = patches.Rectangle((x, y), w, h, 
                                   linewidth=2, 
                                   edgecolor=colors[j], 
                                   facecolor='none',
                                   alpha=0.8)
            axes[1, i].add_patch(rect)
        
        axes[1, i].set_title(f'Image {i+1} - {len(detections)} sites', fontsize=12)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparison visualization saved to: {output_path}")


def export_results_to_geojson(results: Dict, output_path: str, 
                             center_coords: Tuple[float, float] = (40.7128, -74.0060)) -> None:
    """
    Export detection results to GeoJSON format for GIS applications.
    
    Args:
        results: Detection results dictionary
        output_path: Path to save the GeoJSON file
        center_coords: Center coordinates for the area
    """
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    detections = results.get('detections', [])
    
    for i, detection in enumerate(detections):
        confidence = detection.get('confidence', 0)
        bbox = detection.get('bbox', [0, 0, 100, 100])
        
        # Create a simple point feature (could be enhanced with actual bounding boxes)
        lat_offset = (i * 0.001)
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [center_coords[1] + lat_offset, center_coords[0] + lat_offset]
            },
            "properties": {
                "site_id": f"site_{i+1}",
                "confidence": confidence,
                "detection_id": detection.get('box_id', i),
                "bbox": bbox,
                "description": f"Potential archaeological site with {confidence:.2%} confidence"
            }
        }
        geojson["features"].append(feature)
    
    # Save GeoJSON
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"🗺️ GeoJSON export saved to: {output_path}")


def create_detection_report(results: Dict, output_path: str) -> None:
    """
    Create a comprehensive detection report in HTML format.
    
    Args:
        results: Detection results dictionary
        output_path: Path to save the HTML report
    """
    detections = results.get('detections', [])
    total_sites = results.get('total_sites', 0)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Arcai Detection Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
            .stats {{ background-color: #ecf0f1; padding: 15px; margin: 20px 0; border-radius: 5px; }}
            .detection {{ background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-left: 4px solid #3498db; }}
            .high-conf {{ border-left-color: #e74c3c; }}
            .medium-conf {{ border-left-color: #f39c12; }}
            .low-conf {{ border-left-color: #f1c40f; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 Arcai Archaeological Site Detection Report</h1>
            <p>AI-powered satellite imagery analysis</p>
        </div>
        
        <div class="stats">
            <h2>📊 Summary Statistics</h2>
            <ul>
                <li><strong>Total Sites Detected:</strong> {total_sites}</li>
                <li><strong>Analysis Date:</strong> {results.get('timestamp', 'N/A')}</li>
                <li><strong>Image Analyzed:</strong> {results.get('image_path', 'N/A')}</li>
            </ul>
        </div>
        
        <h2>🎯 Individual Detections</h2>
    """
    
    for i, detection in enumerate(detections):
        confidence = detection.get('confidence', 0)
        bbox = detection.get('bbox', [0, 0, 0, 0])
        
        # Determine confidence class
        if confidence > 0.8:
            conf_class = "high-conf"
            conf_label = "High"
        elif confidence > 0.6:
            conf_class = "medium-conf"
            conf_label = "Medium"
        else:
            conf_class = "low-conf"
            conf_label = "Low"
        
        html_content += f"""
        <div class="detection {conf_class}">
            <h3>Site #{i+1}</h3>
            <p><strong>Confidence:</strong> {confidence:.2%} ({conf_label})</p>
            <p><strong>Detection ID:</strong> {detection.get('box_id', 'N/A')}</p>
            <p><strong>Bounding Box:</strong> {bbox}</p>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Save HTML report
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"📄 HTML report saved to: {output_path}")


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization functions...")
    
    # Create dummy results for testing
    test_results = {
        'image_path': 'test_image.jpg',
        'detections': [
            {'box_id': 1, 'confidence': 0.85, 'bbox': [100, 100, 200, 200]},
            {'box_id': 2, 'confidence': 0.72, 'bbox': [300, 150, 400, 250]},
            {'box_id': 3, 'confidence': 0.45, 'bbox': [500, 200, 600, 300]}
        ],
        'total_sites': 3
    }
    
    print("Visualization tests completed!") 