# Project description

In this project, I tried classifying images of space objects into 5 different categories:
1) constellations
2) galaxies
3) nebulas
4) planets
5) stars

I used a convolutional neural network (CNN) in Python.

## 1. Obtaining data and data preprocessing

I obtained an image dataset of space photos on Keggle: https://www.kaggle.com/datasets/abhikalpsrivastava15/space-images-category.
I decided to remove the "cosmos space" folder because those images contain multiple different space objects.

The steps I took for data preprocessing are:
1) Resizing images to a consistent resolution to ensure that all of the images that I feed into a CNN are the same size
2) Normalization, i.e. scaling pixel values to a specific range in order to stabilize training
3) Data Augmentation to increase dataset diversity (random rotations, flips and transformations)
4) Converting RGBA photos to RGB

## 2. Splitting the dataset

We need three sets for CNN training: 
1) training set (for training a machine learning model) - I used 70% of the images
2) validation set (for monitoring training progress and preventing overfitting) - 15% of the images
3) test set (for testing the overall performance of a model) - 15% of the images

## 3. Building a model

The CNN model is specifically used for classifying images. It has different layers:

1) Input layer accepts images of a fixed size (in this project 224x224 pixels)
2) Convolutional layers for extracting features
3) ReLU activation function after each convolutional layer adds non-linearity so the network can learn complex patterns
4) Max-Pooling layer after the group of convolutional layers for better model efficiency
5) Flatten layer reshapes the 2D feature map into a 1D vector
6) Fully connected Dense layers (each neuron is connected to every neuron in previous and subsequent layers). For regularization, a dropout is applied.
7) Output layer produces class probabilities for each space object category

The model is compiled with categorical cross-entropy loss function and I chose the Adam optimizer as the evaluation metric.










