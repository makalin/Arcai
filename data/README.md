# Arcai Dataset

This directory contains satellite imagery data for training and testing the Arcai archaeological site detection model.

## Directory Structure

```
data/
├── positive/          # Images containing archaeological sites
├── negative/          # Images without archaeological sites
├── sample_images/     # Example images for testing
└── README.md         # This file
```

## Data Format

### Supported Image Formats
- **GeoTIFF** (.tif, .tiff) - Preferred for satellite imagery with geospatial metadata
- **JPEG** (.jpg, .jpeg) - Standard image format
- **PNG** (.png) - Lossless image format

### Image Requirements
- **Resolution**: Minimum 224x224 pixels (model input size)
- **Channels**: RGB (3 channels) or multi-spectral (4+ channels)
- **Bit Depth**: 8-bit or 16-bit per channel

### Data Organization

#### Training Data
For training the model, organize your data as follows:

```
data/
├── positive/
│   ├── site_001.tif
│   ├── site_002.tif
│   └── ...
└── negative/
    ├── no_site_001.tif
    ├── no_site_002.tif
    └── ...
```

#### Positive Samples
Images in the `positive/` directory should contain:
- Visible archaeological structures
- Crop marks or soil marks indicating buried features
- Geometric patterns suggesting human activity
- Ancient settlements or ruins

#### Negative Samples
Images in the `negative/` directory should contain:
- Natural landscapes without archaeological features
- Modern urban areas
- Agricultural fields without visible structures
- Water bodies or forested areas

## Data Sources

### Public Domain Satellite Imagery
- **Landsat**: NASA/USGS satellite imagery
- **Sentinel**: European Space Agency satellite data
- **USGS Earth Explorer**: High-resolution aerial imagery

### Archaeological Site Databases
- **OpenStreetMap**: Community-contributed archaeological sites
- **National Heritage Databases**: Government-maintained site registries
- **Academic Publications**: Published archaeological surveys

## Data Preprocessing

The Arcai preprocessing pipeline automatically:
1. **Normalizes** pixel values to [0, 1] range
2. **Resizes** images to model input size (224x224)
3. **Enhances** contrast using histogram equalization
4. **Applies** data augmentation during training

## Usage

### Training
```bash
python src/train.py --data-dir data/ --model-type v1 --epochs 100
```

### Detection
```bash
python src/detect.py --input data/sample_images/test_image.tif
```

## Data Quality Guidelines

### For Best Results:
1. **Use high-resolution imagery** (1-5m per pixel)
2. **Ensure good contrast** between features and background
3. **Include diverse environments** (desert, forest, urban, rural)
4. **Balance positive/negative samples** (aim for 1:1 ratio)
5. **Validate annotations** with archaeological experts

### Common Issues to Avoid:
- **Low-resolution images** that lose detail
- **Poor lighting conditions** that obscure features
- **Seasonal variations** that change appearance
- **Overlapping features** that confuse the model

## Contributing Data

To contribute to the Arcai dataset:

1. **Organize** images into positive/negative directories
2. **Document** the source and location of each image
3. **Include** metadata about archaeological features
4. **Validate** annotations with domain experts
5. **Follow** the naming conventions above

## License

All data in this directory should be:
- **Public domain** or **Creative Commons licensed**
- **Properly attributed** to original sources
- **Compliant** with local regulations

---

For questions about data format or contribution guidelines, please open an issue in the main repository. 