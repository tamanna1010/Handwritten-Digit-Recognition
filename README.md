# Handwritten Digit Recognition

A deep learning project that recognizes handwritten digits (0-9) using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

## Project Overview

This project uses the MNIST dataset to train a CNN model that can accurately predict handwritten digits. The model achieves high accuracy on both training and test datasets.

## Features

- **Dataset**: MNIST (60,000 training images, 10,000 test images)
- **Model Architecture**: Convolutional Neural Network with multiple layers
- **Framework**: TensorFlow/Keras
- **Input Size**: 28x28 grayscale images
- **Output**: Digit prediction (0-9)

## Project Structure

```
Handwritten-Digit-Recognition/
├── notebooks/
│   └── source_code.ipynb          # Main notebook with model training
├── model/
│   └── tf-cnn-model.h5            # Trained model (saved)
├── README.md                       # This file
└── requirements.txt                # Project dependencies
```

## Model Architecture

The CNN model consists of:

```
Conv2D (64 filters, 3x3 kernel) → ReLU
Conv2D (32 filters, 3x3 kernel) → ReLU
MaxPooling2D
Conv2D (16 filters, 3x3 kernel) → ReLU
MaxPooling2D
Conv2D (64 filters, 3x3 kernel) → ReLU
MaxPooling2D
Flatten
Dense (128 units) → ReLU
Dense (10 units) → Softmax
```

**Loss Function**: SparseCategoricalCrossentropy  
**Optimizer**: Adam  
**Metrics**: Accuracy

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Handwritten-Digit-Recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Open and run `notebooks/source_code.ipynb` in Jupyter Notebook:

```bash
jupyter notebook notebooks/source_code.ipynb
```

## Results

- **Training Accuracy**: ~97-99%
- **Test Accuracy**: ~95-98%
- **Epochs**: 10

## Dependencies

- Python 3.7+
- TensorFlow 2.11+
- NumPy
- Matplotlib
- Pillow
- OpenCV (for image processing)

See `requirements.txt` for exact versions.

## Files Description

- **source_code.ipynb**: Main Jupyter notebook containing:
  - Data loading and preprocessing
  - Model architecture definition
  - Model training and validation
  - Visualization of results
  - Model saving and loading

- **tf-cnn-model.h5**: Pre-trained model weights

## Key Features

✅ Data normalization (pixel values 0-1)  
✅ Visualization of training accuracy and loss  
✅ Single and batch image predictions  
✅ Model persistence (save/load functionality)  
✅ Grayscale image support (28x28 pixels)

## Future Improvements

- Add data augmentation for better generalization
- Implement dropout layers to reduce overfitting
- Add batch normalization
- Create a web interface for predictions
- Support for handwritten digit images from real-world sources

## Author

Tamanna


**Note**: Ensure your input images are 28x28 grayscale format for optimal predictions.