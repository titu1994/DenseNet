# Dense Net in Keras
DenseNet implementation of the paper [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993v3.pdf) in Keras

Now supports the more efficient DenseNet-BC (DenseNet-Bottleneck-Compressed) networks. Using the DenseNet-BC-190-40 model, 
it obtaines state of the art performance on CIFAR-10 and CIFAR-100

# Architecture
DenseNet is an extention to Wide Residual Networks. According to the paper: <br>
```
The lth layer has l inputs, consisting of the feature maps of all preceding convolutional blocks. 
Its own feature maps are passed on to all L − l subsequent layers. This introduces L(L+1) / 2 connections 
in an L-layer network, instead of just L, as in traditional feed-forward architectures. 
Because of its dense connectivity pattern, we refer to our approach as Dense Convolutional Network (DenseNet).
```

It features several improvements such as :

1. Dense connectivity : Connecting any layer to any other layer.
2. Growth Rate parameter Which dictates how fast the number of features increase as the network becomes deeper.
3. Consecutive functions : BatchNorm - Relu - Conv which is from the Wide ResNet paper and improvement from the ResNet paper.

The Bottleneck - Compressed DenseNets offer further performance benefits, such as reduced number of parameters, with similar or better performance. 

- Take into consideration the DenseNet-100-12 model, with nearly 7 million parameters against with the DenseNet-BC-100-12, with just 0.8 million parameters.
The BC model acheives 4.51 % error in comparison to the original models' 4.10 % error

- The best original model, DenseNet-100-24 (27.2 million parameters) acheives 3.74 % error, whereas the DenseNet-BC-190-40 (25.6 million parameters) acheives
3.46 % error which is a new state of the art performance on CIFAR-10.

Dense Nets have an architecture which can be shown in the following image from the paper: <br>
<img src="https://github.com/titu1994/DenseNet/blob/master/images/dense_net.JPG?raw=true">

# Performance
The accuracy of DenseNet has been provided in the paper, beating all previous benchmarks in CIFAR 10, CIFAR 100 and SVHN <br>
<img src="https://github.com/titu1994/DenseNet/blob/master/images/accuracy_densenet.JPG?raw=true">

# Usage

Import the `densenet.py` script and use the `create_dense_net(...)` method to create a DenseNet model

Examples : 

```
import densenet

# 'th' dim-ordering or 'tf' dim-ordering
image_dim = (3, 32, 32) or image_dim = (32, 32, 3)

model = densenet.create_dense_net(nb_classes=10, img_dim=image_dim, depth=40, growth_rate=12, 
								   bottleneck=True, reduction=0.5)
```

Weights for the DenseNet-40-12 model are provided ([in the release tab](https://github.com/titu1994/DenseNet/releases)) which have been trained on CIFAR 10. Please extract the appropriate weight file and add it to your weights directory. The default one is for Theano Backend with TH dim ordering.

1. Run the cifar10.py script to train the DenseNet 40 model 
2. Comment out the `model.fit_generator(...)` line and uncomment the `model.load_weights("weights/DenseNet-40-12-CIFAR10.h5")` line to test the classification accuracy.

The current classification accuracy of DenseNet-40-12 is 94.74 %, compared to the accuracy given in the paper - 94.76 %.

# Requirements

- Keras
- Theano (tested) / Tensorflow (not tested, weights not provided)
- h5Py
