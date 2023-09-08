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
3) Data Augmentation to increase dataset diversity (random rotations, flips and transformations








