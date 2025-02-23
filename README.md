# Handwritten-Digit-Classification-Model

This project aims to classify handwritten digits (0-9) using a neural network model built with TensorFlow and Keras. The dataset used is the MNIST dataset, a widely-used dataset for image classification tasks. The model leverages a Fully Connected Neural Network architecture with dropout layers to prevent overfitting and improve generalization.

# Project Overview

Dataset: MNIST Handwritten Digits
Model: Fully connected neural network (Dense layers with ReLU activation) with dropout layers
Achieved Accuracy: 97.98% on the test dataset

# Key Features
# 1. Data Preprocessing:
Loaded the MNIST dataset using TensorFlow’s datasets API.
Normalized the image pixel values to the range [0, 1] by dividing by 255.
Reshaped the images to match the input shape required by the neural network.

# 2. Model Architecture:
The neural network consists of three hidden layers:
Layer 1: 256 neurons with ReLU activation.
Layer 2: 128 neurons with ReLU activation.
Layer 3: 64 neurons with ReLU activation.
Output Layer: 10 neurons (one for each digit) with softmax activation to provide a probability distribution.
Dropout layers were added after each hidden layer to prevent overfitting, with a dropout rate of 0.3.

# 3. Model Optimization:
Optimized using the Adam optimizer with a learning rate of 0.001.
The model was trained with categorical crossentropy loss function, suitable for multi-class classification tasks.

# 4. Model Training:
The model was trained for 10 epochs with a batch size of 32 and a validation split of 0.2 to evaluate performance during training.

# 5. Model Evaluation:
The model achieved an impressive 97.98% accuracy on the test dataset, effectively classifying handwritten digits.
