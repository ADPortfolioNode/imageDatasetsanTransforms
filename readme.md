# Image Datasets and Transforms

This repository contains image datasets and transformation functions for image processing tasks. It provides a collection of commonly used datasets and a set of transformation functions to preprocess and augment the images.

## Datasets

- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): A dataset of 60,000 32x32 color images in 10 classes.
- [MNIST](http://yann.lecun.com/exdb/mnist/): A dataset of handwritten digits, consisting of 60,000 training images and 10,000 test images.
- [ImageNet](http://www.image-net.org/): A large-scale dataset of images with 1,000 classes.

## Transforms

- Resize: Resizes the image to a specified size.
- Normalize: Normalizes the image pixel values to a specified range.
- RandomCrop: Randomly crops a portion of the image.
- RandomHorizontalFlip: Randomly flips the image horizontally.
- RandomRotation: Randomly rotates the image by a specified angle.

## Usage

To use the datasets and transforms in your project, you can import the necessary modules and functions:
