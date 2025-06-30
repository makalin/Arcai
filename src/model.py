"""
ArcaiNet - Custom CNN model for archaeological site detection
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


class ArcaiNet:
    """
    ArcaiNet v1 - Custom CNN architecture for archaeological site detection
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _build_model(self):
        """Build the ArcaiNet architecture."""
        
        inputs = layers.Input(shape=self.input_shape)
        
        # Initial convolution block
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Second convolution block
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Third convolution block
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Fourth convolution block
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def summary(self):
        """Print model summary."""
        return self.model.summary()
    
    def fit(self, *args, **kwargs):
        """Train the model."""
        return self.model.fit(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        """Make predictions."""
        return self.model.predict(*args, **kwargs)
    
    def save(self, filepath):
        """Save the model."""
        self.model.save(filepath)
    
    def load_weights(self, filepath):
        """Load pre-trained weights."""
        self.model.load_weights(filepath)


class ArcaiNetV2(ArcaiNet):
    """
    ArcaiNet v2 - Enhanced version with attention mechanisms
    """
    
    def _build_model(self):
        """Build the enhanced ArcaiNet v2 architecture with attention."""
        
        inputs = layers.Input(shape=self.input_shape)
        
        # Initial convolution block with residual connections
        x = self._conv_block(inputs, 32)
        x = self._conv_block(x, 64)
        x = self._conv_block(x, 128)
        x = self._conv_block(x, 256)
        
        # Attention mechanism
        x = self._attention_block(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers with skip connections
        x = self._dense_block(x, 512)
        x = self._dense_block(x, 256)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'f1_score']
        )
        
        return model
    
    def _conv_block(self, x, filters):
        """Convolutional block with residual connection."""
        residual = x
        
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Add residual connection if dimensions match
        if residual.shape[-1] == filters:
            x = layers.Add()([x, residual])
        
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        return x
    
    def _attention_block(self, x):
        """Attention mechanism to focus on important features."""
        # Channel attention
        avg_pool = layers.GlobalAveragePooling2D()(x)
        max_pool = layers.GlobalMaxPooling2D()(x)
        
        shared_dense = layers.Dense(x.shape[-1] // 8, activation='relu')
        avg_out = shared_dense(avg_pool)
        max_out = shared_dense(max_pool)
        
        channel_attention = layers.Add()([avg_out, max_out])
        channel_attention = layers.Dense(x.shape[-1], activation='sigmoid')(channel_attention)
        channel_attention = layers.Reshape((1, 1, x.shape[-1]))(channel_attention)
        
        x = layers.Multiply()([x, channel_attention])
        
        return x
    
    def _dense_block(self, x, units):
        """Dense block with skip connection."""
        residual = x
        
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # Add skip connection if dimensions match
        if residual.shape[-1] == units:
            x = layers.Add()([x, residual])
        
        return x


def create_model(model_type='v1', **kwargs):
    """Factory function to create ArcaiNet models."""
    if model_type == 'v1':
        return ArcaiNet(**kwargs)
    elif model_type == 'v2':
        return ArcaiNetV2(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation
    model = ArcaiNet()
    model.summary() 