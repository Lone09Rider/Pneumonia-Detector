# Pneumonia Detection Project - Configuration & Setup Guide

## Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download from Kaggle:
```
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
```

Extract to: `data/chest_xray/`

Structure:
```
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 3. Run the Project

```bash
# Run Jupyter Notebook for training
jupyter notebook notebooks/Pneumonia_Detection_Challenge.ipynb

# Run Streamlit Web App
streamlit run app/app.py
```

---

## Project Workflow

### Phase 1: Data Exploration (6 Days)
```python
# In Jupyter notebook
# 1. Load dataset
# 2. Visualize samples
# 3. Analyze distribution
# 4. Check for issues
```

### Phase 2: Preprocessing & Augmentation
```python
# 1. Resize images to 224x224
# 2. Normalize pixel values
# 3. Apply augmentation (rotation, shift, zoom)
# 4. Visualize augmented samples
```

### Phase 3: Model Building (5 Days)
```python
# 1. Build custom CNN
# 2. Build transfer learning models (DenseNet, ResNet, EfficientNet)
# 3. Train models with different hyperparameters
# 4. Save best weights
```

### Phase 4: Evaluation (5 Days)
```python
# 1. Evaluate on test set
# 2. Calculate metrics (accuracy, precision, recall, etc.)
# 3. Generate confusion matrix and ROC curve
# 4. Analyze false positives/negatives
# 5. Fine-tune models based on results
```

### Phase 5: Deployment (3 Days)
```bash
# Run Streamlit app
streamlit run app/app.py

# Features:
# - Upload single image
# - Batch prediction
# - View results
# - Download logs
```

---

## Key Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Precision | TP/(TP+FP) | Avoid false positives |
| Recall | TP/(TP+FN) | Catch all positives |
| Specificity | TN/(TN+FP) | True negative rate |
| F1-Score | 2×(Precision×Recall)/(Precision+Recall) | Balance metric |
| ROC-AUC | Area under ROC curve | Threshold independent |
| Sensitivity | TP/(TP+FN) | True positive rate |

---

## Hyperparameter Configurations

### Baseline Configuration
- **Learning Rate:** 1e-4
- **Batch Size:** 32
- **Epochs:** 30
- **Optimizer:** Adam
- **Loss:** Binary Crossentropy

### Fine-tuning Configuration
- **Learning Rate:** 1e-5 (lower for fine-tuning)
- **Batch Size:** 32
- **Epochs:** 20-30
- **Optimizer:** Adam
- **Freeze Layers:** First 80% of base model

### Data Augmentation
```python
rotation_range=20
width_shift_range=0.2
height_shift_range=0.2
shear_range=0.2
zoom_range=0.2
horizontal_flip=True
```

---

## Model Architectures

### 1. Custom CNN
- Input: 224×224×1 (grayscale)
- 4 Conv blocks with max pooling
- Dense layers with dropout
- Output: Binary classification

### 2. DenseNet121 Transfer Learning
- Input: 224×224×3 (RGB)
- Pretrained on ImageNet
- Global average pooling
- Dense layers for classification

### 3. ResNet50 Transfer Learning
- Input: 224×224×3 (RGB)
- 50 layers deep architecture
- Residual connections
- Fine-tunable layers

### 4. EfficientNetB0 Transfer Learning
- Input: 224×224×3 (RGB)
- Lightweight & efficient
- Compound scaling
- Good for deployment

---

## Troubleshooting

### Issue: CUDA Out of Memory
```python
# Reduce batch size
BATCH_SIZE = 16  # Instead of 32

# Or use memory growth
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### Issue: Model Training Slow
```python
# Use pretrained models (transfer learning)
# Reduce image size to 128×128
# Use mixed precision training
```

### Issue: Low Accuracy
```python
# Increase augmentation
# Try transfer learning
# Increase epochs
# Adjust learning rate
```

---

## Commands Reference

```bash
# Virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install packages
pip install -r requirements.txt
pip install package_name

# Jupyter
jupyter notebook
jupyter lab

# Streamlit
streamlit run app/app.py
streamlit run app/app.py --logger.level=debug

# Training
python -c "
from utils.training import build_custom_cnn, compile_model
model = build_custom_cnn()
model = compile_model(model)
print(model.summary())
"
```

---

## Project Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| Data Analysis | 6 days | EDA, Visualization, Analysis |
| Model Building | 5 days | CNN, Transfer Learning, Training |
| Evaluation | 5 days | Metrics, Fine-tuning, Testing |
| Deployment | 3 days | Streamlit, Integration, Testing |
| **Total** | **2 weeks** | **10-day submission deadline** |

---

## Important Notes

1. **Always download the dataset** before running notebooks
2. **Use virtual environment** to avoid package conflicts
3. **Save model weights** during training for future use
4. **Monitor training** with TensorBoard for insights
5. **Test with diverse images** for robustness verification
6. **Keep logs** of all experiments and results

---

For more details, see `README.md`
