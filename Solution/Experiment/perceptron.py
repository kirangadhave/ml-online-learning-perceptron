import numpy as np
import data_extract as de
import random

class Perceptron:
    def __init__(self, lr):
        self.weights = []
        self.bias = []
        self.lr = lr
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.predictions = []
        self.accuracy = 0
        
    def train(self, data, epoch = 1):
        data_mod = data
        for i in range(epoch - 1):
            np.random.shuffle(data)
            data_mod = np.concatenate((data_mod, data), axis = 0) 
        
        self.X_train = data_mod[:,:-1]
        self.y_train = data_mod[:,-1]
        self.init_weights_bias(self.X_train.shape[1])
        
        for index, i in enumerate(self.X_train):
            h = np.inner(i, self.weights) + self.bias
            f = self.y_train[index]
            if (h*f < 0):
                self.weights = self.weights + self.lr*f*i
                self.bias = self.bias + self.lr*f  
        return self.weights, self.bias

    def init_weights_bias(self, cols):
        ran_init = random.uniform(-0.01, 0.01)
        self.weights = np.array([ran_init]*cols)
        self.bias = ran_init
        
    def predict(self, data):
        self.X_test = data[:,:-1]
        self.y_test = data[:, -1]
        
        preds = []
        for x in self.X_test:
            preds.append(np.inner(x, self.weights) + self.bias)
        self.predictions = preds
        self.accuracy = self.calc_accuracy()
        
    def calc_accuracy(self):
        correct = 0
        for i,x in enumerate(self.predictions):
            if x*self.y_test[i] >= 0:
                correct += 1
        return correct/len(self.y_test)*100

train = de.extract('Dataset/CVSplits/training00.data')

p = Perceptron(0.01)

p.train(train,10)
p.predict(train)

print(p.accuracy)