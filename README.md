
# Distracted Driver Detection

This project aims to build a machine learning model to detect distracted drivers using image data. The notebook contains steps to preprocess the data, train a model, and evaluate its performance.

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Requirements](#requirements)
4. [Usage](#usage)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [Results](#results)

---

## Overview

Distracted driving is a major cause of road accidents. This project leverages a deep learning model to classify driver behaviors into categories such as using a phone, texting, eating, or being attentive while driving.

The notebook contains:
- Data preprocessing
- Model training using a deep learning framework
- Performance evaluation on the validation set

---

## Dataset

The dataset includes labeled images of drivers performing various activities. The images are divided into categories representing different types of distractions. Paths and class names are specified in the notebook.

### Data Directory
Ensure the dataset is structured as follows:
```
/data/
    train/
        c0/  # Safe driving
        c1/  # Texting (right hand)
        ...
    test/
```

---

## Requirements

The notebook is implemented in Python and requires the following libraries:
- Pytorch
- OpenCV
- NumPy
- Pandas
- Matplotlib

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## Usage

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Place the dataset in the `/data` folder.
3. Open the notebook `distracted-driver-detection.ipynb` and run all cells sequentially.

---

## Model Training

The model is based on a Convolutional Neural Network (CNN) architecture. It is trained using:
- Data augmentation to improve generalization
- Categorical cross-entropy loss
- Adam optimizer

Adjust hyperparameters in the notebook for experimentation.

---

## Evaluation

The model's performance is evaluated using accuracy and a confusion matrix. Visualization of results includes:
- Sample predictions
- Class-wise performance metrics

---

## Results

The final model achieves the following metrics:
- Accuracy: 99%
- Precision: 99%
- Recall: 99%

Detailed results are available in the notebook's output section.

---

## Contributing

Contributions are welcome. Fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
