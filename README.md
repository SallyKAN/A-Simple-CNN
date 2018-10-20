# A-Simple-CNN

This is a a very simple, three-layer neural network implemented with Pytorch. 

The dataset used here is the CIFAR-10 image dataset. It is spilt into:
* 45000 training images
* 5000 validation imgaes
* 10000 test images

To run this program, you need to install:
* torch
* torchvision
* numpy
* matplotlib

## The Model
The layers of this model are as following:

1. number of Convolutional Layers: 3
2. numebr of ReLU (Rectified Linear Units) Layers: 6
2. number of Pooling Layers: 3
3. number of Fully Connected Layers:3

They are structured as following:
```
INPUT -> [CONV -> RELU -> POOL]*3 -> [FC -> RELU]*3 -> FC -> OUTPUT
```
Where `*` indicates repetition.

The model architecture is highlihted as below:                                    
* [3x32x32] INPUT                                          
* [32x32x32] CONV1 32 5x5 filters at stride = 2, padding = 1       
* [32x16x16] MAX POOL1 2x2 filter at stride = 2
* [64x16x16] CONV2 64 3x3 filters at stride = 1 padding = 1          
* [64x8x8] MAX POOL2 2x2 filter at stride = 2
* [128X8X8] CONV3 128 3x3 filters at stride = 1 padding = 1
* [128x4x4] MAX POPL3 2x2 filter at stride = 2
* [2048] FC1 2048 neurons
* [240] FC2 240 neurons
* [120] FC3 120 neurons
* [10] FC4 neurons (class scores)

## Evaluation
![img](/accr.png)
