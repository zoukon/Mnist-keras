# Mnist-keras
This repository belongs to my masters project. The purpose is to more easily share my code over the different machine learning platforms I investigate. The main purpose of the project has been to survey different machine learning platforms documentation to determine which platforms we wanted to test. Then choose a handful of those to recreate the same experiment and review the results. After the initial survey, the platforms with the best results were: BEAT-EU, Kaggle Kernels, Codalab, OpenML and CometML. The appropriate experiments can be found here:

### BEAT
### Kaggle
https://www.kaggle.com/zoukon/keras2-jupyter/edit/run/12698808
### Codalab
https://worksheets.codalab.org/worksheets/0x32ce25fce564404ca002540190e168c9
### OpenML
https://www.openml.org/f/9704
### CometML
https://www.comet.ml/zoukon/mnist

The experiment itself is a simple Convolutional neural network made with Keras, trained and tested on the Mnist dataset. This is still a work in progress. The code varies slightly from system to system. I have the different files I have uploaded here, to make it more clear which alterations had to be made to the original code to make it run on the different platforms. The platforms that use the appropriate files are as follows:


## Overview
The purpose of this experiment to categorize the MNIST dataset by using a convolutional neural network (CNN). MNIST is a set of 70,000 labeled images 
of handwritten numbers, centered over 28x28 pixels. The purpose of the experiment is to train the neural network to recognize handwritten images 
on this form, and verify the accuracy through the test set. We expect that the network should be able to correctly assess more than 98% 
of the test set after training, similar to results we have gotten while running the experiment locally.
In order to make the CNN, I will be using Tensorflow and Keras, and basing myself on the example presented in 
the tensorflow guide for CNNs. https://www.tensorflow.org/tutorials/estimators/cnn as well as the CNN example from 
https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a .


## CNN
A CNN is a type of deep neural network commonly used in image recognition. Very little preprocessing is usually used compared to other image
classification algorithms. The network typically consists of multiple hidden layers of three different types, each having a different purpose.

#### Convolutional layer: 
This layer applies convolution to the input. 
For each subregion, the layer performs a set of mathematical operations to produce a single value in the output feature map. 
Convolutional layers then typically apply a ReLU activation function to the output to introduce nonlinearities into the model.
#### Pooling layer:
The pooling layers combine the output of neuron clusters at one layer into a single neuron in the next layer. This is handled by a pooling
algorithm such as max, min or average pooling. The main purpose of this layer is to reduce processing time by discarding a set of the values. 
There is usually a pooling layer behind every convolutional layer in the network. 
#### Dense layer:
The dense layers or fully connected layer (FC) perform classification 
on the feature results from extraction and downsampling by the previous layers,
by connecting every neuron in one layer to every neutron in another layer. The principle of this 
layer is similar to a traditional multi-layer perceptron neural network. The dense layers are typically at the end of the network.

## Preprocessing
CNNs traditionally use very little preprocessing of the data. 
In this case we simply import the data as a numpy.ndarray,
normalize the values, and reshape the matrix to pass it into the network. 
We also convert the class vectors to binary class matrices for the classification.

## Method
The first thing we do is to download the data and prepare it, so that we can pass it into the network. The full Mnist dataset is
available through in the keras datasets. In the notebook in the Keras-tensorflow workspace, we downloaded the 

This implementation is built using 2 convolutional layers both followed by a pooling layer. after the convolutions we performed
dropout to improve the models convergence. We then flatten the data to pass it on to the dense layers. Here we have 2 layers,
with a dropout inbetween. The last dense layer uses softmax as its loss function, while all previous layers use rectified linear regression as their 
activation function. 

This leaves us with the following structure of the network: 

`Conv(relu) -> Pool(Max) -> Conv(relu) -> Pool(Max) -> Dropout(0.25) -> Dense(relu) -> Flatten -> Dropout(0.5) -> Dense(softmax)`

For the first convolution, I chose an output space of 32 output, kernel size of 3x3, stride of 1x1 and no padding. 
The second convolution is identical, except it has an output space of 64. 
Pooling layers are both identical, and use MaxPooling with a pool size of 2x2, stride of 2x2 and no padding.

Rough overview of the code layout: 
```
Import Mnist dataset
Reshape arrays
convert class vectors to binary class matrices
Initialize model
Conv(relu) -> Pool(Max) -> Conv(relu) -> Pool(Max) -> 
Dropout(0.25) -> Dense(relu) -> Flatten -> Dropout(0.5) -> Dense(softmax)
Compile model
Evaluate model
Save results to file
Save model as HDF5
```
