"""
Quick Start Script for Pneumonia Detection Project
Run this script to set up and test the project
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_step(text):
    print(f"\n✓ {text}")

def print_warning(text):
    print(f"\n⚠️  {text}")

def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")
    
    required = [
        'tensorflow',
        'keras',
        'numpy',
        'pandas',
        'opencv',
        'sklearn',
        'streamlit'
    ]
    
    try:
        import tensorflow as tf
        print_step(f"TensorFlow {tf.__version__} installed")
    except ImportError:
        print_warning("TensorFlow not installed. Run: pip install -r requirements.txt")
        return False
    
    return True

def check_directories():
    """Check if required directories exist"""
    print_header("Checking Directories")
    
    dirs = [
        'data',
        'notebooks',
        'app',
        'utils',
        'models',
    ]
    
    for dir_name in dirs:
        if os.path.exists(dir_name):
            print_step(f"Directory '{dir_name}/' exists")
        else:
            print_warning(f"Directory '{dir_name}/' not found")

def check_dataset():
    """Check if dataset is downloaded"""
    print_header("Checking Dataset")
    
    dataset_path = "data/chest_xray"
    
    if os.path.exists(dataset_path):
        splits = ['train', 'test', 'val']
        for split in splits:
            split_path = os.path.join(dataset_path, split)
            if os.path.exists(split_path):
                normal = len(os.listdir(os.path.join(split_path, 'NORMAL'))) if os.path.exists(os.path.join(split_path, 'NORMAL')) else 0
                pneumonia = len(os.listdir(os.path.join(split_path, 'PNEUMONIA'))) if os.path.exists(os.path.join(split_path, 'PNEUMONIA')) else 0
                print_step(f"{split.upper()}: {normal} Normal, {pneumonia} Pneumonia")
            else:
                print_warning(f"Split '{split}' not found in dataset")
        return True
    else:
        print_warning(f"Dataset not found at '{dataset_path}'")
        print("  Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        return False

def test_imports():
    """Test if all libraries can be imported"""
    print_header("Testing Imports")
    
    libraries = [
        ('tensorflow', 'TensorFlow'),
        ('keras', 'Keras'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'scikit-learn'),
        ('streamlit', 'Streamlit'),
    ]
    
    failed = []
    for lib, name in libraries:
        try:
            __import__(lib)
            print_step(f"{name} imported successfully")
        except ImportError:
            print_warning(f"Failed to import {name}")
            failed.append(name)
    
    return len(failed) == 0

def display_project_structure():
    """Display project structure"""
    print_header("Project Structure")
    
    structure = """
FP/
├── data/                          # Dataset directory
│   └── chest_xray/               # Images (to be downloaded)
│
├── notebooks/                     # Jupyter notebooks
│   └── Pneumonia_Detection_Challenge.ipynb
│
├── app/                          # Streamlit application
│   └── app.py
│
├── utils/                        # Utility modules
│   ├── utilities.py
│   └── training.py
│
├── models/                       # Trained models (created during training)
│   └── logs/
│
├── requirements.txt              # Dependencies
├── README.md                     # Full documentation
└── SETUP_GUIDE.md               # Setup instructions
    """
    
    print(structure)

def main():
    """Main function"""
    
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       PNEUMONIA DETECTION CHALLENGE - Quick Start            ║")
    print("║                                                              ║")
    print("║  A Healthcare AI Project for Pneumonia Detection in X-rays  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    # Check everything
    display_project_structure()
    check_directories()
    deps_ok = check_dependencies()
    dataset_ok = check_dataset()
    imports_ok = test_imports()
    
    print_header("Next Steps")
    
    if not dataset_ok:
        print("""
1. Download the dataset:
   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
   
2. Extract to: data/chest_xray/
   Structure should be:
   ├── train/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   ├── test/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   └── val/
       ├── NORMAL/
       └── PNEUMONIA/
        """)
    
    if not deps_ok or not imports_ok:
        print("""
Install dependencies:
   pip install -r requirements.txt
        """)
    
    if deps_ok and dataset_ok:
        print("""
All checks passed! You can now:

1. Run the Jupyter Notebook:
   jupyter notebook notebooks/Pneumonia_Detection_Challenge.ipynb
   
2. Or run the Streamlit Web App:
   streamlit run app/app.py

3. For detailed documentation, see:
   - README.md (Full project documentation)
   - SETUP_GUIDE.md (Setup and configuration)
        """)
    
    print("\n" + "="*60)
    print("  Project Ready! Happy Building! 🚀")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
