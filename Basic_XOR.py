from matplotlib.pylab import randint
import numpy as np

# XOR problem. Points (0,0), (0, 1), (1, 0), (1, 1) 
# Where the first and fourth are classified as 0 and the second and third are classified as 1
Inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
Outputs = [0, 1, 1, 0]

W11 = 0.5
W12 = 0.5
W21 = 0.5
W22 = 0.5
b1 = 0
b2 = 0

W31 = 0.5
W32 = 0.5
b3 = 0

learning_rate = 0.1
epochs = 50000

for epoch in range(epochs):
    #Forward Pass (go through the network and make predictions for a single sample from the dataset)
    Input = Inputs[randint(0, 3)] #Select a random point and set it as the input. We will only be working with 1 point at a time
    h1 = Input[0] * W11 + Input[1] * W21 + b1
    h2 = Input[0] * W12 + Input[1] * W22 + b2

    output_predicted = h1 * W31 + h2 * W32 + b3

    #backpropagation

    
