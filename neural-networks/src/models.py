import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# Please read the free response questions before starting to code.

class Digit_Classifier(nn.Module):
    """
    This is the class that creates a neural network for classifying handwritten digits
    from the MNIST dataset.
	
	Network architecture:
	- Input layer
	- First hidden layer: fully connected layer of size 128 nodes
	- Second hidden layer: fully connected layer of size 64 nodes
	- Output layer: a linear layer with one node per class (in this case 10)

	Activation function: ReLU for both hidden layers

    """
    def __init__(self):
        super(Digit_Classifier, self).__init__()
        self.first_layer = nn.Linear(28 * 28, 128)
        self.second_layer = nn.Linear(128, 64)
        self.third_layer = nn.Linear(64, 10)

    def forward(self, input):
        input = F.relu(self.first_layer(input))
        input = F.relu(self.second_layer(input))
        input = self.third_layer(input)
        return input
    

class Dog_Classifier_FC(nn.Module):
    """
    This is the class that creates a fully connected neural network for classifying dog breeds
    from the DogSet dataset.
    
    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size 128 nodes
    - Second hidden layer: fully connected layer of size 64 nodes
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers

    """

    def __init__(self):
        super(Dog_Classifier_FC, self).__init__()
        self.first_layer = nn.Linear(64 * 64 * 3, 128)
        self.second_layer = nn.Linear(128, 64)
        self.third_layer = nn.Linear(64, 10)

    def forward(self, input):
        input = F.relu(self.first_layer(input))
        input = F.relu(self.second_layer(input))
        input = self.third_layer(input)
        return input


class Dog_Classifier_Conv(nn.Module):
    """
    This is the class that creates a convolutional neural network for classifying dog breeds
    from the DogSet dataset.
    
    Network architecture:
    - Input layer
    - First hidden layer: convolutional layer of size (select kernel size and stride)
    - Second hidden layer: convolutional layer of size (select kernel size and stride)
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers
    
    There should be a maxpool after each convolution. 
    
    The sequence of operations looks like this:
    
    	1. Apply convolutional layer with stride and kernel size specified
		- note: uses hard-coded in_channels and out_channels
		- read the problems to figure out what these should be!
	2. Apply the activation function (ReLU)
	3. Apply 2D max pooling with a kernel size of 2

    Inputs: 
    kernel_size: list of length 2 containing kernel sizes for the two convolutional layers
                 e.g., kernel_size = [(3,3), (3,3)]
    stride: list of length 2 containing strides for the two convolutional layers
            e.g., stride = [(1,1), (1,1)]

    """

    def __init__(self, kernel_size, stride):
        super(Dog_Classifier_Conv, self).__init__()
        self.convolutional_layer1 = nn.Conv2d(3, 16, kernel_size[0], stride[0])
        self.convolutional_layer2 = nn.Conv2d(16, 32, kernel_size[1], stride[1])
        self.first_layer = nn.Linear(32 * 13 * 13, 10)   

    def forward(self, input):
        input = input.permute(0,3,1,2)
        input = F.max_pool2d(F.relu(self.convolutional_layer1(input)), 2)
        input = F.max_pool2d(F.relu(self.convolutional_layer2(input)), 2)
        input = input.view(input.size(0), -1)
        input = self.first_layer(input)
        return input


class Synth_Classifier(nn.Module):
    """
    This is the class that creates a convolutional neural network for classifying 
    synthesized images.
    
    Network architecture:
    - Input layer
    - First hidden layer: convolutional layer of size (select kernel size and stride)
    - Second hidden layer: convolutional layer of size (select kernel size and stride)
    - Output layer: a linear layer with one node per class (in this case 2)

    Activation function: ReLU for both hidden layers
    
    There should be a maxpool after each convolution. 
    
    The sequence of operations looks like this:
    
    	1. Apply convolutional layer with stride and kernel size specified
		- note: uses hard-coded in_channels and out_channels
		- read the problems to figure out what these should be!
	2. Apply the activation function (ReLU)
	3. Apply 2D max pooling with a kernel size of 2

    Inputs: 
    kernel_size: list of length 3 containing kernel sizes for the three convolutional layers
                 e.g., kernel_size = [(5,5), (3,3),(3,3)]
    stride: list of length 3 containing strides for the three convolutional layers
            e.g., stride = [(1,1), (1,1),(1,1)]

    """

    def __init__(self, kernel_size, stride):
        super(Synth_Classifier, self).__init__()   
        self.convolutional_layer1 = nn.Conv2d(1, 2, kernel_size[0], stride[0])
        self.convolutional_layer2 = nn.Conv2d(2, 4, kernel_size[1], stride[1])
        self.convolutional_layer3 = nn.Conv2d(4, 8, kernel_size[2], stride[2])
        self.first_layer = nn.Linear(8 * 1 * 1, 2)
        
    def forward(self, input):
        input = input.permute(0, 3, 1, 2)
        input = F.max_pool2d(F.relu(self.convolutional_layer1(input)), 2)
        input = F.max_pool2d(F.relu(self.convolutional_layer2(input)), 2)
        input = F.max_pool2d(F.relu(self.convolutional_layer3(input)), 2)
        input = input.view(-1, 8 * 1 * 1)
        input = self.first_layer(input)
        return input














