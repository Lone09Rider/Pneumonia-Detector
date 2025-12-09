"""
Training module for Pneumonia Detection
Contains model definitions and training functions
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121, ResNet50, EfficientNetB0
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
import os


def build_custom_cnn(input_shape=(224, 224, 1)):
    """Build custom CNN model"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def build_densenet_transfer(input_shape=(224, 224, 3)):
    """Build DenseNet121 transfer learning model"""
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False
    
    model = models.Sequential([
        layers.Input(shape=input_shape),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def build_resnet_transfer(input_shape=(224, 224, 3)):
    """Build ResNet50 transfer learning model"""
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False
    
    model = models.Sequential([
        layers.Input(shape=input_shape),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def build_efficientnet_transfer(input_shape=(224, 224, 3)):
    """Build EfficientNetB0 transfer learning model"""
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False
    
    model = models.Sequential([
        layers.Input(shape=input_shape),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def compile_model(model, learning_rate=1e-4):
    """Compile model with standard settings"""
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    return model


def create_callbacks(model_name="model", save_dir="../models"):
    """Create training callbacks"""
    os.makedirs(save_dir, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(save_dir, f'{model_name}_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(save_dir, 'logs'),
            histogram_freq=1
        )
    ]
    return callbacks


def train_model(model, train_generator, val_generator=None, 
                epochs=30, callbacks=None, model_name="model", save_dir="../models"):
    """Train the model"""
    if callbacks is None:
        callbacks = create_callbacks(model_name, save_dir)
    
    if val_generator is not None:
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            train_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    
    return history


def fine_tune_model(model, unfreeze_layers=50, learning_rate=1e-5):
    """Fine-tune transfer learning model by unfreezing layers"""
    # Unfreeze layers
    for layer in model.layers[-unfreeze_layers:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model


# Configuration for different model experiments
MODELS_CONFIG = {
    "custom_cnn": {
        "builder": build_custom_cnn,
        "input_shape": (224, 224, 1),
        "description": "Custom CNN from scratch"
    },
    "densenet121": {
        "builder": build_densenet_transfer,
        "input_shape": (224, 224, 3),
        "description": "DenseNet121 with transfer learning"
    },
    "resnet50": {
        "builder": build_resnet_transfer,
        "input_shape": (224, 224, 3),
        "description": "ResNet50 with transfer learning"
    },
    "efficientnet_b0": {
        "builder": build_efficientnet_transfer,
        "input_shape": (224, 224, 3),
        "description": "EfficientNetB0 with transfer learning"
    }
}


# Hyperparameter presets
HYPERPARAMETERS = {
    "baseline": {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 30
    },
    "aggressive": {
        "learning_rate": 1e-3,
        "batch_size": 16,
        "epochs": 50
    },
    "conservative": {
        "learning_rate": 1e-5,
        "batch_size": 64,
        "epochs": 20
    }
}


if __name__ == "__main__":
    print("Pneumonia Detection - Model Training Module")
    print("=" * 60)
    
    print("\nAvailable Models:")
    for name, config in MODELS_CONFIG.items():
        print(f"  - {name}: {config['description']}")
    
    print("\nAvailable Hyperparameter Presets:")
    for name in HYPERPARAMETERS.keys():
        print(f"  - {name}")
    
    print("\n✓ Ready for training")
