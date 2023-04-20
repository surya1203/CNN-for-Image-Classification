# Cifar-10 Dataset and CNN Model

This repository contains an implementation of Convolutional Neural Network (CNN) using the Cifar-10 dataset. The code loads the Cifar-10 dataset using Keras and Tensorflow libraries, applies preprocessing, and trains a CNN model using the training dataset. The trained model is then evaluated on the testing dataset, and the performance is visualized through loss curves.

## Requirements
- keras
- tensorflow
- numpy
- matplotlib

## Installation
To install the required libraries, run the following command:
```
pip install keras tensorflow numpy matplotlib
```

## Dataset
The code loads the Cifar-10 dataset using Keras and Tensorflow libraries. The dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 testing images.

## CNN Model
The CNN model used in this implementation has the following layers:
- Conv2D layer with 32 filters and (3, 3) kernel size, ReLU activation, and input shape of (32, 32, 3)
- MaxPooling2D layer with (2, 2) pool size
- Conv2D layer with 64 filters and (4, 4) kernel size, ReLU activation
- MaxPooling2D layer with (2, 2) pool size
- Flatten layer
- Dense layer with 256 neurons and ReLU activation
- Dropout layer with 0.3 rate
- Dense layer with 10 neurons and softmax activation

## Training and Evaluation
The model is trained on the training dataset using Adadelta optimizer and categorical_crossentropy loss function. The model is trained for 200 epochs with a batch size of 16, and the loss curves are plotted for both training and validation datasets. Then, the model is recompiled with the same hyperparameters and evaluated on the testing dataset. The testing accuracy and loss are printed in the console.

## How to Use
To run the code, execute the `cifar10_cnn.py` file. This will load the dataset, preprocess the data, train the CNN model, evaluate the model, and plot the loss curves. The trained model will be saved in the working directory as `cifar10_cnn_model.h5`. The saved model can be loaded using `keras.models.load_model` function and used for prediction.

## Credits
The code in this repository is based on the following resources:
- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
