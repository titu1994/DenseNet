# Dense Net in Keras
DenseNet implementation of the paper [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993v1.pdf) in Keras

# Architecture
DenseNet is an extention to Wide Residual Networks. According to the paper: <br>
```
the lth layer has l inputs, consisting of the feature maps of all preceding convolutional blocks. 
Its own feature maps are passed on to all L − l subsequent layers. This introduces L(L+1) / 2 connections 
in an L-layer network, instead of just L, as in traditional feed-forward architectures. 
Because of its dense connectivity pattern, we refer to our approach as Dense Convolutional Network (DenseNet).
```

It features several improvements such as :

1. Dense connectivity : connecting any layer to any other layer.
2. Growth Rate parameter which dictates how fast the number of features increase as the network becomes deeper.
3. Consecutive functions : BatchNorm - Relu - Conv which is from the Wide ResNet paper.

Dense Nets have an architecture which can be shown in the following image from the paper: <br>
<img src="https://github.com/titu1994/DenseNet/blob/master/images/dense_net.JPG?raw=true">

# Performance
The accuracy of DenseNet has been provided in the paper, beating all previous benchmarks in CIFAR 10, CIFAR 100 and SVHN <br>
<img src="https://github.com/titu1994/DenseNet/blob/master/images/accuracy_densenet.JPG?raw=true">

# Usage

Provided are the weights for CIFAR 10 with the DenseNet 40 model.

1. Run the cifar10.py script to train the DenseNet 40 model 
2. Comment out the `model.fit_generator(...)` line and uncomment the `model.load_weights("weights/DenseNet-40-12-CIFAR10.h5")` line to test the classification accuracy.

# Requirements

- Keras
- Theano (tested) / Tensorflow (not tested, weights not provided)
- h5Py
