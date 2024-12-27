# Medical Image Classification: Detecting Pediatric Pneumonia

## Overview
This project uses a convolutional neural network (CNN) to classify chest X-ray images as either **Normal** or indicating **Pneumonia** (bacterial or viral). It is aimed at beginners interested in medical image classification and explores fundamental steps such as data preprocessing, augmentation, and training a model with a pre-trained CNN architecture.

---

## Dataset

The dataset contains chest X-ray images organized into three folders:
- `train/`: Training images
- `val/`: Validation images
- `test/`: Testing images

The dataset is imbalanced and requires techniques to address this issue during model training.

---

## Features

- **Dataset Preprocessing**: Load, resize, normalize, and assign labels to images.
- **Image Augmentation**: Artificially increase dataset size using transformations (e.g., rotation, zoom, shear, horizontal flip).
- **Model Training**: Utilizes MobileNet, a lightweight pre-trained CNN architecture, with class weights to handle imbalance.
- **Visualization**: Plots training and validation accuracy and loss curves to evaluate model performance.

---

## Requirements

To run the code, you need the following:

- Python 3.7+
- Libraries:
  - `tensorflow`
  - `keras`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `opencv-python`
  - `Pillow`

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## Steps to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/medical-image-classification.git
   cd medical-image-classification
   ```

2. **Prepare the Dataset**:
   - Download the chest X-ray dataset and organize it into `train/`, `val/`, and `test/` folders.
   - Update the `base_dir` path in the code to point to your dataset location.

3. **Run the Script**:
   Execute the Python script:
   ```bash
   python main.py
   ```

4. **Model Training and Evaluation**:
   - The script preprocesses images, trains the CNN model, and evaluates it on the test set.
   - Training and validation metrics are plotted to assess performance.

---

## Key Highlights

### Data Preprocessing
- Resizes images to 224x224 pixels.
- Normalizes pixel values.
- Handles class imbalance by computing class weights.

### Training Details
- **Batch Size**: 32
- **Architecture**: MobileNet (lightweight CNN)
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Binary Accuracy and Mean Absolute Error
- **Epochs**: 10 (modifiable)

### Results Visualization
- Training vs. Validation Accuracy
- Training vs. Validation Loss

---

## Visualization

Sample chest X-ray images:

| Normal | Pneumonia |
|--------|-----------|
| ![Normal Sample](images/normal_sample.png) | ![Pneumonia Sample](images/pneumonia_sample.png) |

Class distribution across training, validation, and test sets:

![Class Distribution](images/class_distribution.png)

Training and validation accuracy/loss curves:

![Training vs. Validation Accuracy](images/training_validation_accuracy.png)

---

## Future Enhancements

- Implement additional data augmentation techniques.
- Experiment with different pre-trained architectures (e.g., ResNet, EfficientNet).
- Fine-tune the model with transfer learning.
- Deploy the model using Flask or FastAPI for real-time inference.

---



