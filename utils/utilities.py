"""
Utility functions for Pneumonia Detection Project
"""

import cv2
import numpy as np
from pathlib import Path
import os
import json
from datetime import datetime
import tensorflow as tf


class ImageProcessor:
    """Image preprocessing utilities"""
    
    def __init__(self, img_size=224):
        self.img_size = img_size
    
    def load_image(self, image_path):
        """Load image from file"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return image
    
    def resize_image(self, image):
        """Resize image to target size"""
        return cv2.resize(image, (self.img_size, self.img_size))
    
    def normalize_image(self, image):
        """Normalize image pixel values"""
        return image.astype(np.float32) / 255.0
    
    def apply_clahe(self, image):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply((image * 255).astype(np.uint8))
    
    def preprocess(self, image, apply_clahe=False):
        """Complete preprocessing pipeline"""
        # Resize
        image = self.resize_image(image)
        
        # Optional CLAHE
        if apply_clahe:
            image = self.apply_clahe(image)
        
        # Normalize
        image = self.normalize_image(image)
        
        # Add batch and channel dimensions
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        
        return image


class PredictionLogger:
    """Log and manage predictions"""
    
    def __init__(self, log_dir="../logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "predictions_log.json")
    
    def log_prediction(self, filename, prediction, confidence, threshold):
        """Log a prediction"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "prediction": prediction,
            "confidence": float(confidence),
            "threshold": float(threshold)
        }
        
        # Load existing logs
        logs = self._load_logs()
        logs.append(log_entry)
        
        # Save logs
        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def _load_logs(self):
        """Load existing logs"""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return []
    
    def get_statistics(self):
        """Get prediction statistics"""
        logs = self._load_logs()
        
        if not logs:
            return None
        
        total = len(logs)
        pneumonia = sum(1 for log in logs if log["prediction"] == "PNEUMONIA")
        normal = total - pneumonia
        
        avg_confidence = np.mean([log["confidence"] for log in logs])
        
        return {
            "total_predictions": total,
            "pneumonia_cases": pneumonia,
            "normal_cases": normal,
            "average_confidence": avg_confidence
        }


class ModelEvaluator:
    """Model evaluation utilities"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_pred_probs=None):
        """Calculate evaluation metrics"""
        from sklearn.metrics import (
            confusion_matrix, accuracy_score, precision_score,
            recall_score, f1_score, roc_auc_score
        )
        
        # Basic metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Rates
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        
        # AUC-ROC
        auc_roc = None
        if y_pred_probs is not None:
            auc_roc = roc_auc_score(y_true, y_pred_probs)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "fpr": fpr,
            "fnr": fnr,
            "auc_roc": auc_roc,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        }
    
    @staticmethod
    def threshold_analysis(y_true, y_pred_probs, thresholds=None):
        """Analyze performance across different thresholds"""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_probs > threshold).astype(int)
            
            results.append({
                "threshold": threshold,
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "f1_score": f1_score(y_true, y_pred)
            })
        
        return results


class DataAugmenter:
    """Data augmentation utilities"""
    
    @staticmethod
    def apply_rotation(image, angle_range=20):
        """Apply random rotation"""
        angle = np.random.uniform(-angle_range, angle_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))
    
    @staticmethod
    def apply_flip(image, horizontal=True):
        """Apply flip augmentation"""
        if horizontal:
            return cv2.flip(image, 1)
        else:
            return cv2.flip(image, 0)
    
    @staticmethod
    def apply_noise(image, noise_type='gaussian'):
        """Add noise to image"""
        if noise_type == 'gaussian':
            noise = np.random.normal(0, 0.01, image.shape)
            return np.clip(image + noise, 0, 1)
        return image
    
    @staticmethod
    def apply_brightness(image, factor_range=(0.8, 1.2)):
        """Adjust brightness"""
        factor = np.random.uniform(*factor_range)
        return np.clip(image * factor, 0, 1)


# Usage examples
if __name__ == "__main__":
    print("Pneumonia Detection Utilities")
    print("=" * 50)
    
    # Example: Image processing
    print("\n1. Image Processing Example:")
    processor = ImageProcessor(img_size=224)
    print("   - Image Processor initialized")
    
    # Example: Prediction logging
    print("\n2. Prediction Logging Example:")
    logger = PredictionLogger()
    logger.log_prediction("sample.jpg", "PNEUMONIA", 0.89, 0.5)
    print("   - Sample prediction logged")
    
    # Example: Model evaluation
    print("\n3. Model Evaluation Example:")
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
    print(f"   - Accuracy: {metrics['accuracy']:.4f}")
    
    print("\n✓ All utilities ready for use")
