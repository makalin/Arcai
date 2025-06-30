#!/usr/bin/env python3
"""
Arcai - Model Training Script
Train the ArcaiNet model on archaeological site detection data.
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ArcaiNet, ArcaiNetV2
from src.preprocessing import preprocess_image, create_training_batch
from src.visualization import plot_training_history


class ArcaiDataGenerator:
    """Custom data generator for Arcai training data."""
    
    def __init__(self, image_paths, labels, batch_size=32, target_size=(224, 224), augment=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.index = 0
    
    def __len__(self):
        return len(self.image_paths) // self.batch_size
    
    def __next__(self):
        if self.index >= len(self.image_paths):
            self.index = 0
            # Shuffle data
            indices = np.random.permutation(len(self.image_paths))
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
        
        batch_paths = self.image_paths[self.index:self.index + self.batch_size]
        batch_labels = self.labels[self.index:self.index + self.batch_size]
        
        self.index += self.batch_size
        
        # Create batch
        images, labels = create_training_batch(
            batch_paths, batch_labels, len(batch_paths), 
            self.target_size, self.augment
        )
        
        return images, labels


def load_training_data(data_dir: str) -> tuple:
    """
    Load training data from directory structure.
    
    Expected structure:
    data_dir/
    ├── positive/  # Images with archaeological sites
    └── negative/  # Images without archaeological sites
    
    Args:
        data_dir: Path to training data directory
    
    Returns:
        Tuple of (image_paths, labels)
    """
    image_paths = []
    labels = []
    
    # Load positive samples
    positive_dir = os.path.join(data_dir, 'positive')
    if os.path.exists(positive_dir):
        for img_file in os.listdir(positive_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                image_paths.append(os.path.join(positive_dir, img_file))
                labels.append(1)
    
    # Load negative samples
    negative_dir = os.path.join(data_dir, 'negative')
    if os.path.exists(negative_dir):
        for img_file in os.listdir(negative_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                image_paths.append(os.path.join(negative_dir, img_file))
                labels.append(0)
    
    print(f"📊 Loaded {len(image_paths)} training samples:")
    print(f"   - Positive samples: {sum(labels)}")
    print(f"   - Negative samples: {len(labels) - sum(labels)}")
    
    return image_paths, labels


def split_data(image_paths: list, labels: list, train_ratio: float = 0.8, 
               val_ratio: float = 0.1) -> tuple:
    """
    Split data into train/validation/test sets.
    
    Args:
        image_paths: List of image paths
        labels: List of labels
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Shuffle data
    indices = np.random.permutation(len(image_paths))
    shuffled_paths = [image_paths[i] for i in indices]
    shuffled_labels = [labels[i] for i in indices]
    
    # Calculate split indices
    n_total = len(image_paths)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split data
    train_paths = shuffled_paths[:n_train]
    train_labels = shuffled_labels[:n_train]
    
    val_paths = shuffled_paths[n_train:n_train + n_val]
    val_labels = shuffled_labels[n_train:n_train + n_val]
    
    test_paths = shuffled_paths[n_train + n_val:]
    test_labels = shuffled_labels[n_train + n_val:]
    
    print(f"📈 Data split:")
    print(f"   - Training: {len(train_paths)} samples")
    print(f"   - Validation: {len(val_paths)} samples")
    print(f"   - Test: {len(test_paths)} samples")
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def create_callbacks(model_dir: str, patience: int = 10) -> list:
    """
    Create training callbacks.
    
    Args:
        model_dir: Directory to save models
        patience: Patience for early stopping
    
    Returns:
        List of callbacks
    """
    callbacks = [
        # Model checkpoint
        ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks


def train_model(model, train_data, val_data, callbacks, epochs=100, batch_size=32):
    """
    Train the model.
    
    Args:
        model: ArcaiNet model instance
        train_data: Training data tuple (paths, labels)
        val_data: Validation data tuple (paths, labels)
        callbacks: Training callbacks
        epochs: Number of training epochs
        batch_size: Batch size
    
    Returns:
        Training history
    """
    train_paths, train_labels = train_data
    val_paths, val_labels = val_data
    
    # Create data generators
    train_generator = ArcaiDataGenerator(train_paths, train_labels, batch_size, augment=True)
    val_generator = ArcaiDataGenerator(val_paths, val_labels, batch_size, augment=False)
    
    # Train model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, test_data, batch_size=32):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained model
        test_data: Test data tuple (paths, labels)
        batch_size: Batch size
    
    Returns:
        Evaluation results
    """
    test_paths, test_labels = test_data
    test_generator = ArcaiDataGenerator(test_paths, test_labels, batch_size, augment=False)
    
    # Evaluate model
    results = model.evaluate(test_generator, verbose=1)
    
    # Create results dictionary
    metrics = ['loss', 'accuracy', 'precision', 'recall']
    evaluation_results = dict(zip(metrics, results))
    
    print("📊 Test Results:")
    for metric, value in evaluation_results.items():
        print(f"   - {metric.capitalize()}: {value:.4f}")
    
    return evaluation_results


def save_training_config(config: dict, output_path: str):
    """Save training configuration to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Arcai - Model Training')
    parser.add_argument('--data-dir', '-d', required=True, help='Training data directory')
    parser.add_argument('--model-type', '-m', default='v1', choices=['v1', 'v2'], 
                       help='Model type (v1 or v2)')
    parser.add_argument('--output-dir', '-o', default='models', help='Output directory')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', '-p', type=int, default=10, help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load training data
    print("🔄 Loading training data...")
    image_paths, labels = load_training_data(args.data_dir)
    
    if len(image_paths) == 0:
        print("❌ No training data found!")
        return 1
    
    # Split data
    print("📊 Splitting data...")
    train_data, val_data, test_data = split_data(image_paths, labels)
    
    # Create model
    print(f"🏗️ Creating ArcaiNet {args.model_type}...")
    if args.model_type == 'v1':
        model = ArcaiNet()
    else:
        model = ArcaiNetV2()
    
    # Create callbacks
    callbacks = create_callbacks(args.output_dir, args.patience)
    
    # Training configuration
    config = {
        'model_type': args.model_type,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'patience': args.patience,
        'data_dir': args.data_dir,
        'training_date': datetime.now().isoformat(),
        'total_samples': len(image_paths),
        'positive_samples': sum(labels),
        'negative_samples': len(labels) - sum(labels)
    }
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'training_config.json')
    save_training_config(config, config_path)
    
    # Train model
    print("🚀 Starting training...")
    history = train_model(model, train_data, val_data, callbacks, 
                         args.epochs, args.batch_size)
    
    # Evaluate model
    print("📊 Evaluating model...")
    evaluation_results = evaluate_model(model, test_data, args.batch_size)
    
    # Save evaluation results
    eval_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(eval_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Plot training history
    history_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history, history_path)
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, f'arcainet_{args.model_type}_final.h5')
    model.save(final_model_path)
    
    print(f"✅ Training completed! Model saved to: {final_model_path}")
    print(f"📊 Results saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 