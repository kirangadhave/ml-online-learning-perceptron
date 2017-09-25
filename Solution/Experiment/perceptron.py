import numpy as np
import data_extract as de
from random_seed import random_seed
import random

#To change seed value, change the value of random_seed variable in file 'random_seed.py'
# Uncomment the below line to seed the random functions. 

random.seed(random_seed)

class Perceptron:
    def __init__(self, margin = 0):
        self.weights = []
        self.bias = []
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.predictions = []
        self.accuracy = 0
        self.init_weights_bias(68)
        self.updates = 0
        self.t = 0
        self.margin = margin
        self.av_weights = self.weights
        self.av_bias = self.bias
        
    def train(self, data, lr, dynamic_lr = False, aggressive = False):
        if aggressive:
            self.train_aggressive(data, lr)
        else:
            data_mod = data
            lr_0 = lr
            self.X_train = data_mod[:,:-1]
            self.y_train = data_mod[:,-1]
    #        self.init_weights_bias(self.X_train.shape[1])
            
            for index, i in enumerate(self.X_train):
                h = np.inner(i, self.weights) + self.bias
                f = self.y_train[index]
                if dynamic_lr:
                        lr = lr_0/(1+self.t)
                        self.t += 1
                if (h*f <= self.margin):
                    self.weights = self.weights + lr*f*i
                    self.bias = self.bias + lr*f
                    self.updates += 1    
                self.av_weights = self.av_weights + self.weights
                self.av_bias = self.av_bias + self.bias
            
    def train_aggressive(self, data, margin):
        data_mod = np.c_[np.ones((data.shape[0],1)), data]
        self.init_weights_bias(69)
        self.X_train = data_mod[:,:-1]
        self.y_train = data_mod[:,-1]
#        self.init_weights_bias(self.X_train.shape[1])
        for index, i in enumerate(self.X_train):
            h = np.inner(i, self.weights) + self.bias
            f = self.y_train[index]
            if (h*f <= margin):
                lr = (margin - f*np.inner(self.weights, i))/(np.inner(i, i) + 1)
                self.weights = self.weights + lr*f*i
                self.bias = self.bias + lr*f
                self.updates += 1
#            self.av_weights = self.av_weights + self.weights
#            self.av_bias = self.av_bias + self.bias
        
    def init_weights_bias(self, cols):
        ran_init = random.uniform(-0.01, 0.01)
        self.weights = np.array([ran_init]*cols)
        self.bias = ran_init
        
    def predict(self, data, average = False, aggressive = False):
        if aggressive:
            self.predict_aggressive(data)
        else:
            self.X_test = data[:,:-1]
            self.y_test = data[:, -1]
            w = self.weights
            b = self.bias
            if average:
                w = self.av_weights
                b = self.av_bias
                
            preds = []
            for x in self.X_test:
                preds.append(np.inner(x, w) + b)
            self.predictions = preds
            self.accuracy = self.calc_accuracy()
            
    def predict_aggressive(self, data):
        data = np.c_[np.ones((data.shape[0],1)), data]
        self.X_test = data[:,:-1]
        self.y_test = data[:, -1]
        w = self.weights
        b = self.bias
                    
        preds = []
        for x in self.X_test:
            preds.append(np.inner(x, w) + b)
        self.predictions = preds
        self.accuracy = self.calc_accuracy()
        
    def calc_accuracy(self):
        correct = 0
        for i,x in enumerate(self.predictions):
            if x*self.y_test[i] >= 0:
                correct += 1
        return correct/len(self.y_test)*100
#
#train = de.extract(['Dataset/CVSplits/training00.data', 'Dataset/CVSplits/training01.data', 'Dataset/CVSplits/training02.data', 'Dataset/CVSplits/training03.data'])
#test = de.extract(['Dataset/CVSplits/training04.data'])
#p = Perceptron()
#p.train(train, 0.01)
#p.predict(train)
#print(p.accuracy)