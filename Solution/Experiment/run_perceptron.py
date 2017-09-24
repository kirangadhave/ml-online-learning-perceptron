import numpy as np
import data_extract as de
from perceptron import Perceptron
import copy

train1, train2, train3, train4, train5, test1, test2, test3, test4, test5 = [],[],[],[],[],[],[],[],[],[]

def set_cross_validation():
    global train1, train2, train3, train4, train5, test1, test2, test3, test4, test5
    
    train1  = de.extract(['Dataset/CVSplits/training00.data', 'Dataset/CVSplits/training01.data', 'Dataset/CVSplits/training02.data', 'Dataset/CVSplits/training03.data'])
    test1   = de.extract(['Dataset/CVSplits/training04.data'])
    train2  = de.extract(['Dataset/CVSplits/training01.data', 'Dataset/CVSplits/training02.data', 'Dataset/CVSplits/training03.data', 'Dataset/CVSplits/training04.data'])
    test2   = de.extract(['Dataset/CVSplits/training00.data'])
    train3  = de.extract(['Dataset/CVSplits/training02.data', 'Dataset/CVSplits/training03.data', 'Dataset/CVSplits/training04.data', 'Dataset/CVSplits/training00.data'])
    test3   = de.extract(['Dataset/CVSplits/training01.data'])
    train4  = de.extract(['Dataset/CVSplits/training03.data', 'Dataset/CVSplits/training04.data', 'Dataset/CVSplits/training00.data', 'Dataset/CVSplits/training01.data'])
    test4   = de.extract(['Dataset/CVSplits/training02.data'])
    train5  = de.extract(['Dataset/CVSplits/training04.data', 'Dataset/CVSplits/training00.data', 'Dataset/CVSplits/training01.data', 'Dataset/CVSplits/training02.data'])
    test5   = de.extract(['Dataset/CVSplits/training03.data'])
    
set_cross_validation()

def simple_perceptron():
    global train1, train2, train3, train4, train5, test1, test2, test3, test4, test5
    learning_rates = [1,0.1,0.01]
    
    acc_dictionary = {}
    
    #Cross Validation
    for lr in learning_rates:
        epoch = 10
        p1 = Perceptron()
        p2 = Perceptron()
        p3 = Perceptron()
        p4 = Perceptron()
        p5 = Perceptron()
                
        for x in range(epoch):
            np.random.shuffle(train1)
            np.random.shuffle(train2)
            np.random.shuffle(train3)
            np.random.shuffle(train4)
            np.random.shuffle(train5)
            
            p1.train(train1, lr)
            p2.train(train2, lr)
            p3.train(train3, lr)
            p4.train(train4, lr)
            p5.train(train5, lr)
        
        p1.predict(test1)
        p2.predict(test2)
        p3.predict(test3)
        p4.predict(test4)
        p5.predict(test5)
        
        acc_dictionary[lr] = (p1.accuracy + p1.accuracy + p1.accuracy + p1.accuracy + p1.accuracy)/5
    
    best_hp = max(acc_dictionary, key=acc_dictionary.get)
    
    #Final Run
    train = de.extract(["Dataset/phishing.train"])
    dev = de.extract(["Dataset/phishing.dev"])
    test = de.extract(["Dataset/phishing.test"])
    
    epoch = 20
    
    epoch_acc_dict = []
    p = Perceptron()
    
    for i in range(epoch):
        p.train(train, best_hp)
        p.predict(dev)
        epoch_acc_dict.append((copy.deepcopy(p), p.accuracy))
    print(epoch_acc_dict)
    P = max(epoch_acc_dict, key = lambda x:x[1])
    print(P)
    
simple_perceptron()