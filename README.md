# Convolutional Neural Network (CNN) on MNIST Dataset

## Objective

This project demonstrates how to implement a Convolutional Neural Network (CNN) using TensorFlow to classify handwritten digits from the MNIST dataset. 
The MNIST dataset consists of 60,000 training images and 10,000 testing images of handwritten digits from 0 to 9.

## Requirements

To run this project with CPU only, you need the following dependencies installed:

    Python 3.12.2+
    TensorFlow
    NumPy
    Matplotlib
    Sklearn
    io
    itertools

To run this project with compatible GPU natively on Windows, you need the following dependencies installed:

    Python 3.10.14
    Tensorflow
    Cudatoolkit 11.2
    Cudnn 8.1.0
    NumPy
    Matplotlib
    Sklearn
    io
    itertools

## Results

After running the script, you can expect the model to achieve an accuracy of over 99% on the test set. 
The exact accuracy may vary depending on the training conditions and random initialisation.

![Confusion Matrix](https://github.com/JackXLin/CNN_with_MNIST_dataset/blob/main/Capture.JPG)

Above Confusion Matrix shows model's performance on individual digit, while most digit have 100% accuracy the digit 5 have 98% and it's mostly confused with number 3
This is perhaps due to both 3 and 5 have a similar bottom half curvature and needs to be invetigated further.