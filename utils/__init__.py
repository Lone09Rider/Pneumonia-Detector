"""
Pneumonia Detection - Utility Modules
Package for image processing, training, and evaluation utilities
"""

from .utilities import (
    ImageProcessor,
    PredictionLogger,
    ModelEvaluator,
    DataAugmenter
)

from .training import (
    build_custom_cnn,
    build_densenet_transfer,
    build_resnet_transfer,
    build_efficientnet_transfer,
    compile_model,
    create_callbacks,
    train_model,
    fine_tune_model,
    MODELS_CONFIG,
    HYPERPARAMETERS
)

__version__ = "1.0.0"
__author__ = "Pneumonia Detection Project"

__all__ = [
    # Utilities
    "ImageProcessor",
    "PredictionLogger",
    "ModelEvaluator",
    "DataAugmenter",
    
    # Training
    "build_custom_cnn",
    "build_densenet_transfer",
    "build_resnet_transfer",
    "build_efficientnet_transfer",
    "compile_model",
    "create_callbacks",
    "train_model",
    "fine_tune_model",
    "MODELS_CONFIG",
    "HYPERPARAMETERS",
]
