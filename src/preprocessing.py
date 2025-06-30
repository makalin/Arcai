"""
Image preprocessing utilities for Arcai
"""

import cv2
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
import os
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess satellite image for model input.
    
    Args:
        image_path: Path to the satellite image
        target_size: Target size for the model (height, width)
    
    Returns:
        Preprocessed image as numpy array
    """
    # Try to load as raster first (for GeoTIFF files)
    try:
        with rasterio.open(image_path) as src:
            # Read all bands
            image = src.read()
            # Convert to RGB format (assuming 3+ bands)
            if image.shape[0] >= 3:
                image = np.transpose(image[:3], (1, 2, 0))
            else:
                # Single band - convert to 3-channel
                image = np.repeat(image[0][..., np.newaxis], 3, axis=-1)
    except:
        # Fallback to PIL for other formats
        image = np.array(Image.open(image_path))
        if len(image.shape) == 2:
            image = np.repeat(image[..., np.newaxis], 3, axis=-1)
    
    # Normalize to [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    
    # Apply histogram equalization for better contrast
    image = enhance_contrast(image)
    
    # Resize to target size
    image = resize_image(image, target_size)
    
    return image


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast using histogram equalization.
    
    Args:
        image: Input image array
    
    Returns:
        Enhanced image array
    """
    if len(image.shape) == 3:
        # Apply CLAHE to each channel
        enhanced = np.zeros_like(image)
        for i in range(image.shape[2]):
            enhanced[:, :, i] = cv2.equalizeHist((image[:, :, i] * 255).astype(np.uint8)) / 255.0
        return enhanced
    else:
        return cv2.equalizeHist((image * 255).astype(np.uint8)) / 255.0


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to target size using bilinear interpolation.
    
    Args:
        image: Input image array
        target_size: Target size (height, width)
    
    Returns:
        Resized image array
    """
    return cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)


def augment_image(image: np.ndarray, augmentation_type: str = 'random') -> np.ndarray:
    """
    Apply data augmentation to image.
    
    Args:
        image: Input image array
        augmentation_type: Type of augmentation ('random', 'flip', 'rotate', 'brightness')
    
    Returns:
        Augmented image array
    """
    if augmentation_type == 'random':
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = np.fliplr(image)
        
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            image = rotate_image(image, angle)
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            image = adjust_brightness(image, factor)
    
    elif augmentation_type == 'flip':
        image = np.fliplr(image)
    
    elif augmentation_type == 'rotate':
        angle = np.random.uniform(-30, 30)
        image = rotate_image(image, angle)
    
    elif augmentation_type == 'brightness':
        factor = np.random.uniform(0.7, 1.3)
        image = adjust_brightness(image, factor)
    
    return image


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image by given angle.
    
    Args:
        image: Input image array
        angle: Rotation angle in degrees
    
    Returns:
        Rotated image array
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    return rotated


def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Adjust image brightness.
    
    Args:
        image: Input image array
        factor: Brightness factor (>1 for brighter, <1 for darker)
    
    Returns:
        Brightness-adjusted image array
    """
    adjusted = image * factor
    return np.clip(adjusted, 0, 1)


def extract_patches(image: np.ndarray, patch_size: Tuple[int, int], stride: int) -> List[np.ndarray]:
    """
    Extract patches from image for sliding window analysis.
    
    Args:
        image: Input image array
        patch_size: Size of patches (height, width)
        stride: Stride for patch extraction
    
    Returns:
        List of image patches
    """
    patches = []
    height, width = image.shape[:2]
    patch_h, patch_w = patch_size
    
    for y in range(0, height - patch_h + 1, stride):
        for x in range(0, width - patch_w + 1, stride):
            patch = image[y:y + patch_h, x:x + patch_w]
            patches.append(patch)
    
    return patches


def normalize_spectral_bands(image: np.ndarray) -> np.ndarray:
    """
    Normalize spectral bands for satellite imagery.
    
    Args:
        image: Input image with spectral bands
    
    Returns:
        Normalized image array
    """
    if len(image.shape) == 3 and image.shape[2] > 3:
        # Multi-spectral image - normalize each band
        normalized = np.zeros_like(image)
        for i in range(image.shape[2]):
            band = image[:, :, i]
            band_min, band_max = np.min(band), np.max(band)
            if band_max > band_min:
                normalized[:, :, i] = (band - band_min) / (band_max - band_min)
            else:
                normalized[:, :, i] = band
        return normalized
    else:
        return image


def apply_ndvi_filter(image: np.ndarray, red_band: int = 2, nir_band: int = 3) -> np.ndarray:
    """
    Apply NDVI (Normalized Difference Vegetation Index) filter.
    
    Args:
        image: Input image with spectral bands
        red_band: Index of red band (0-based)
        nir_band: Index of near-infrared band (0-based)
    
    Returns:
        NDVI filtered image
    """
    if len(image.shape) == 3 and image.shape[2] > max(red_band, nir_band):
        red = image[:, :, red_band]
        nir = image[:, :, nir_band]
        
        # Calculate NDVI
        ndvi = (nir - red) / (nir + red + 1e-8)
        
        # Create RGB visualization
        ndvi_rgb = np.zeros((*ndvi.shape, 3))
        ndvi_rgb[:, :, 0] = np.clip((1 - ndvi) * 2, 0, 1)  # Red channel
        ndvi_rgb[:, :, 1] = np.clip(ndvi * 2, 0, 1)       # Green channel
        ndvi_rgb[:, :, 2] = 0                             # Blue channel
        
        return ndvi_rgb
    else:
        return image


def create_training_batch(image_paths: List[str], labels: List[int], 
                         batch_size: int, target_size: Tuple[int, int] = (224, 224),
                         augment: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a training batch from image paths and labels.
    
    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels
        batch_size: Size of the batch
        target_size: Target image size
        augment: Whether to apply augmentation
    
    Returns:
        Tuple of (images, labels) arrays
    """
    batch_images = []
    batch_labels = []
    
    for i in range(min(batch_size, len(image_paths))):
        # Load and preprocess image
        image = preprocess_image(image_paths[i], target_size)
        
        # Apply augmentation if requested
        if augment:
            image = augment_image(image)
        
        batch_images.append(image)
        batch_labels.append(labels[i])
    
    return np.array(batch_images), np.array(batch_labels)


if __name__ == "__main__":
    # Test preprocessing functions
    print("Testing preprocessing functions...")
    
    # Create a dummy image for testing
    test_image = np.random.rand(512, 512, 3).astype(np.float32)
    
    # Test resize
    resized = resize_image(test_image, (224, 224))
    print(f"Resized shape: {resized.shape}")
    
    # Test augmentation
    augmented = augment_image(test_image, 'random')
    print(f"Augmented shape: {augmented.shape}")
    
    print("Preprocessing tests completed!") 