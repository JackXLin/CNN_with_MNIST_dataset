
# Convolutional Neural Network (CNN) on MNIST Dataset

## Objective

This project demonstrates how to implement a Convolutional Neural Network (CNN) using TensorFlow to classify handwritten digits from the MNIST dataset. The MNIST dataset consists of 60,000 training images and 10,000 testing images of handwritten digits from 0 to 9.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Notebook Content](#notebook-content)
- [Results](#results)
- [Contributing](#contributing)
- [Licence](#licence)

## Introduction

The MNIST dataset is a collection of 70,000 handwritten digits commonly used for training various image processing systems. This project showcases a step-by-step implementation of a CNN to classify the MNIST digits.

## Requirements

To run this project with CPU only, you need the following dependencies installed:

- Python 3.12.2+
- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn
- io
- itertools

To run this project with a compatible GPU natively on Windows, you need the following dependencies installed:

- Python 3.10.14
- TensorFlow
- Cudatoolkit 11.2
- Cudnn 8.1.0
- NumPy
- Matplotlib
- Scikit-learn
- io
- itertools

## Installation

To run the notebook, you need to have Python and the following packages installed:

- TensorFlow
- TensorFlow Datasets
- NumPy
- Matplotlib
- Scikit-learn

You can install the required packages using the following command:

```bash
pip install tensorflow tensorflow-datasets numpy matplotlib scikit-learn
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/JackXLin/CNN_with_MNIST_dataset.git
    cd CNN_MNIST
    ```

2. Open the Jupyter Notebook:
    ```bash
    jupyter notebook CNN_MNIST.ipynb
    ```

3. Execute the cells in the notebook to train and evaluate the CNN model on the MNIST dataset.

## Notebook Content

The notebook includes the following steps:

1. **Import Libraries**: Import necessary libraries including TensorFlow, TensorFlow Datasets, NumPy, Matplotlib, and Scikit-learn.

2. **Data Preparation**: Load and preprocess the MNIST dataset, including scaling the images and splitting the data into training, validation, and test sets.

3. **Model Building**: Define the CNN architecture using TensorFlow's Keras API.

4. **Training**: Train the CNN model on the training set and validate it using the validation set.

5. **Evaluation**: Evaluate the trained model on the test set and visualise the results using a confusion matrix.

## Results

After running the script, you can expect the model to achieve an accuracy of over 99% on the test set. The exact accuracy may vary depending on the training conditions and random initialisation.

![Confusion Matrix](https://github.com/JackXLin/CNN_with_MNIST_dataset/blob/main/Capture.JPG)

The above confusion matrix shows the model's performance on individual digits. While most digits have 100% accuracy, the digit 5 has 98% and is mostly confused with the number 3. This is perhaps due to both 3 and 5 having a similar bottom half curvature and needs to be investigated further.

![Hyperparameters](https://github.com/JackXLin/CNN_with_MNIST_dataset/blob/main/Hyperparameter_tuning.JPG)

The above table shows different combinations of hyperparameters and their accuracy.
