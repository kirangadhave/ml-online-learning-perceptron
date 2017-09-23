import numpy as np
import data_extract as de
class Perceptron:
    def __init__(self, lr):
        self.weights = []
        self.bias = []
        self.lr = lr
        
    def train(data, epoch = 1):
        data_mod = np.ndarray([])
        for i in range(epoch):
            data_mod = np.append(data_mod, np.random.shuffle(data))
        return data_mod
#        x = data[:,:-1]
#        y = data[-1]


a = de.extract('Dataset/CVSplits/training00.data')
