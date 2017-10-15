import numpy as np
import math


class Neuron(object):

    def __init__(self, input_size):
        self.W = np.random.randn(input_size)
        self.bias = 1

    def forward(self, inputs):
        cell_body_sum = np.sum(inputs * self.W + self.bias)
        firing_rate = 1.0 / (1.0 + math.exp(-cell_body_sum)) # sigmoid activation function
        return firing_rate
