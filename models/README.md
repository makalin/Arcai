# Arcai Models

This directory contains pre-trained ArcaiNet models for archaeological site detection.

## Model Files

### ArcaiNet v1
- **File**: `arcainet_v1.h5` (or `arcainet_v1_final.h5`)
- **Architecture**: Custom CNN with 4 convolutional blocks
- **Input**: 224x224x3 RGB images
- **Output**: Binary classification (site/no site)
- **Use case**: General archaeological site detection

### ArcaiNet v2
- **File**: `arcainet_v2.h5` (or `arcainet_v2_final.h5`)
- **Architecture**: Enhanced CNN with attention mechanisms and residual connections
- **Input**: 224x224x3 RGB images
- **Output**: Binary classification (site/no site)
- **Use case**: High-accuracy detection with attention to important features

## Model Performance

### ArcaiNet v1 Performance
- **Accuracy**: ~85-90%
- **Precision**: ~82-88%
- **Recall**: ~87-92%
- **Training Time**: ~2-4 hours (100 epochs)

### ArcaiNet v2 Performance
- **Accuracy**: ~88-93%
- **Precision**: ~85-90%
- **Recall**: ~90-94%
- **Training Time**: ~3-6 hours (100 epochs)

## Usage

### Loading a Model
```python
import tensorflow as tf
from src.model import ArcaiNet, ArcaiNetV2

# Load ArcaiNet v1
model = tf.keras.models.load_model('models/arcainet_v1.h5')

# Or create and load weights
model = ArcaiNet()
model.load_weights('models/arcainet_v1_weights.h5')
```

### Making Predictions
```python
from src.preprocessing import preprocess_image
import numpy as np

# Load and preprocess image
image = preprocess_image('path/to/satellite_image.tif')

# Make prediction
prediction = model.predict(np.expand_dims(image, axis=0))
confidence = prediction[0][0]

print(f"Archaeological site detected with {confidence:.2%} confidence")
```

## Training Configuration

Each model comes with a training configuration file (`training_config.json`) that includes:
- Model architecture parameters
- Training hyperparameters
- Dataset information
- Training date and duration

## Model Validation

Models are validated on:
- **Test set**: 10% of training data
- **Cross-validation**: 5-fold cross-validation
- **External validation**: Independent archaeological datasets

## Model Updates

### Version History
- **v1.0**: Initial release with basic CNN architecture
- **v1.1**: Improved data augmentation and regularization
- **v2.0**: Added attention mechanisms and residual connections
- **v2.1**: Enhanced preprocessing pipeline

### Updating Models
To update a model:
1. Retrain with new data
2. Validate on test set
3. Compare performance metrics
4. Update version number
5. Document changes

## Model Limitations

### Known Limitations
- **Resolution dependency**: Works best with 1-5m resolution imagery
- **Environmental factors**: Performance varies with vegetation cover and season
- **Site type specificity**: Optimized for settlement and structural remains
- **Geographic bias**: Trained primarily on Mediterranean and Near Eastern sites

### Recommendations
- Use high-resolution satellite imagery
- Apply during dry seasons for better visibility
- Validate detections with ground truth data
- Consider local archaeological context

## File Structure

```
models/
├── arcainet_v1.h5              # Pre-trained ArcaiNet v1
├── arcainet_v1_final.h5        # Final trained v1 model
├── arcainet_v2.h5              # Pre-trained ArcaiNet v2
├── arcainet_v2_final.h5        # Final trained v2 model
├── training_config.json        # Training configuration
├── evaluation_results.json     # Model evaluation metrics
├── training_history.json       # Training history data
└── README.md                   # This file
```

## Contributing

To contribute new models:
1. Train model using the provided training script
2. Validate on test dataset
3. Document model architecture and performance
4. Include training configuration and results
5. Update this README with new model information

## License

Models are released under the same license as the main project (MIT License).

---

For questions about model usage or performance, please refer to the main documentation or open an issue. 