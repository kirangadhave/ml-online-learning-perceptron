import numpy as np
import data_extract as de
import random

class Perceptron:
    def __init__(self, lr):
        self.weights = []
        self.bias = []
        self.lr = lr
        
    def train(self, data, epoch = 1):
        data_mod = data
        for i in range(epoch - 1):
            np.random.shuffle(data)
            data_mod = np.concatenate((data_mod, data), axis = 0) 
        
        x = data_mod[:,:-1]
        y = data_mod[:,-1]
        self.init_weights_bias(x.shape[1])
        
        for index, i in enumerate(x):
            h = np.inner(i, self.weights) + self.bias
            f = y[index]
            if (h*f < 0):
                self.weights = self.weights + self.lr*f*i
                self.bias = self.bias + self.lr*f
        return self.weights, self.bias

    def init_weights_bias(self, cols):
        ran_init = random.uniform(-0.01, 0.01)
        self.weights = np.array([ran_init]*cols)
        self.bias = ran_init
        
        
    
a = de.extract('Dataset/CVSplits/training00.data')

p = Perceptron(1)

a = p.train(a,10)

