import numpy as np
import data_extract as de
from perceptron import Perceptron
import copy
import matplotlib.pyplot as plt
import time

plt.ioff()
plt.close("all")
train1, train2, train3, train4, train5, test1, test2, test3, test4, test5 = [],[],[],[],[],[],[],[],[],[]

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

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
    
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

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
        
        acc_dictionary[lr] = (p1.accuracy + p2.accuracy + p3.accuracy + p4.accuracy + p5.accuracy)/5
    
    best_hp = max(acc_dictionary, key=acc_dictionary.get)
    
    #Final Run
    train = de.extract(["Dataset/phishing.train"])
    dev = de.extract(["Dataset/phishing.dev"])
    test = de.extract(["Dataset/phishing.test"])
    
    epoch = 20
    
    epoch_acc_dict = []
    p = Perceptron()
        
    for i in range(epoch):
        np.random.shuffle(train)
        p.train(train, best_hp)
        p.predict(dev)
        epoch_acc_dict.append((copy.deepcopy(p), p.accuracy))
    
    P_set = max(epoch_acc_dict, key = lambda x:x[1])
    P = P_set[0]
    P.predict(test)
    
#   
#    print(P.accuracy)
    print("Best learning rate = " + str(best_hp))
    print("Cross validation accuracy for best learning rate = " + str(acc_dictionary[best_hp]))
    print("Total number of updates performed by the learning algorithm on training set = " + str(P.updates))
    print("Developement set accuracy = " + str(P_set[1]))
    print("Test set accuracy = " + str(P.accuracy))
    
    x_axis = list(range(1,21))
    y_axis = [x[1] for x in epoch_acc_dict]
    plt.plot(x_axis, y_axis)
    plt.ylim([1,100])
    
#     Uncomment the below line to see the plot
#    plt.show()
    
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
    
def dynamic_perceptron():
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
                     
            p1.train(train1, lr, True)
            p2.train(train2, lr, True)
            p3.train(train3, lr, True)
            p4.train(train4, lr, True)
            p5.train(train5, lr, True)
        
        p1.predict(test1)
        p2.predict(test2)
        p3.predict(test3)
        p4.predict(test4)
        p5.predict(test5)
        
        acc_dictionary[lr] = (p1.accuracy + p2.accuracy + p3.accuracy + p4.accuracy + p5.accuracy)/5
    
    best_hp = max(acc_dictionary, key=acc_dictionary.get)
    
    #Final Run
    train = de.extract(["Dataset/phishing.train"])
    dev = de.extract(["Dataset/phishing.dev"])
    test = de.extract(["Dataset/phishing.test"])
    
    epoch = 20
    
    epoch_acc_dict = []
    p = Perceptron()
        
    for i in range(epoch):
        np.random.shuffle(train)
        p.train(train, best_hp, True)
        p.predict(dev)
        epoch_acc_dict.append((copy.deepcopy(p), p.accuracy))
    
    P_set = max(epoch_acc_dict, key = lambda x:x[1])
    P = P_set[0]
    P.predict(test)
    
#   
#    print(P.accuracy)
    print("Best learning rate = " + str(best_hp))
    print("Cross validation accuracy for best learning rate = " + str(acc_dictionary[best_hp]))
    print("Total number of updates performed by the learning algorithm on training set = " + str(P.updates))
    print("Developement set accuracy = " + str(P_set[1]))
    print("Test set accuracy = " + str(P.accuracy))
    
    x_axis = list(range(1,21))
    y_axis = [x[1] for x in epoch_acc_dict]
    plt.plot(x_axis, y_axis)
    plt.ylim([1,100])
    
#     Uncomment the below line to see the plot
#    plt.show()

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

def margin_perceptron():
    global train1, train2, train3, train4, train5, test1, test2, test3, test4, test5
        
    learning_rates = [1,0.1,0.01]
    margins = [1, 0.1, 0.01]
    
    combinations = [(x,y) for x in margins for y in learning_rates]
    
    acc_dictionary = []
    
    #Cross Validation
    for c in combinations:
        epoch = 10
        p1 = Perceptron(c[0])
        p2 = Perceptron(c[0])
        p3 = Perceptron(c[0])
        p4 = Perceptron(c[0])
        p5 = Perceptron(c[0])
        
        for x in range(epoch):
            np.random.shuffle(train1)
            np.random.shuffle(train2)
            np.random.shuffle(train3)
            np.random.shuffle(train4)
            np.random.shuffle(train5)
                     
            p1.train(train1, c[1], True)
            p2.train(train2, c[1], True)
            p3.train(train3, c[1], True)
            p4.train(train4, c[1], True)
            p5.train(train5, c[1], True)
        
        p1.predict(test1)
        p2.predict(test2)
        p3.predict(test3)
        p4.predict(test4)
        p5.predict(test5)
        
        acc_dictionary.append((c,(p1.accuracy + p2.accuracy + p3.accuracy + p4.accuracy + p5.accuracy)/5))
    
    best_hp_set = max(acc_dictionary, key = lambda x:x[1])
    best_hp = best_hp_set[0]
    #Final Run
    train = de.extract(["Dataset/phishing.train"])
    dev = de.extract(["Dataset/phishing.dev"])
    test = de.extract(["Dataset/phishing.test"])
    
    epoch = 20
    
    epoch_acc_dict = []
    p = Perceptron(best_hp[0])
        
    for i in range(epoch):
        np.random.shuffle(train)
        p.train(train, best_hp[1], True)
        p.predict(dev)
        epoch_acc_dict.append((copy.deepcopy(p), p.accuracy))
    
    P_set = max(epoch_acc_dict, key = lambda x:x[1])
    P = P_set[0]
    P.predict(test)
    
#   
#    print(P.accuracy)
    print("Best learning rate = " + str(best_hp[1]))
    print("Best margin = " + str(best_hp[0]))
    print("Cross validation accuracy for best learning rate = " + str(best_hp_set[1]))
    print("Total number of updates performed by the learning algorithm on training set = " + str(P.updates))
    print("Developement set accuracy = " + str(P_set[1]))
    print("Test set accuracy = " + str(P.accuracy))
    
    x_axis = list(range(1,21))
    y_axis = [x[1] for x in epoch_acc_dict]
    plt.plot(x_axis, y_axis)
    plt.ylim([1,100])
    
#     Uncomment the below line to see the plot
#    plt.show()

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
    
def average_perceptron():
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
        
        p1.predict(test1, True)
        p2.predict(test2, True)
        p3.predict(test3, True)
        p4.predict(test4, True)
        p5.predict(test5, True)
        
        acc_dictionary[lr] = (p1.accuracy + p2.accuracy + p3.accuracy + p4.accuracy + p5.accuracy)/5
    
    best_hp = max(acc_dictionary, key=acc_dictionary.get)
    
    #Final Run
    train = de.extract(["Dataset/phishing.train"])
    dev = de.extract(["Dataset/phishing.dev"])
    test = de.extract(["Dataset/phishing.test"])
    
    epoch = 20
    
    epoch_acc_dict = []
    p = Perceptron()
        
    for i in range(epoch):
        np.random.shuffle(train)
        p.train(train, best_hp)
        p.predict(dev, True)
        epoch_acc_dict.append((copy.deepcopy(p), p.accuracy))
    
    P_set = max(epoch_acc_dict, key = lambda x:x[1])
    P = P_set[0]
    P.predict(test, True)
    
#   
#    print(P.accuracy)
    print("Best learning rate = " + str(best_hp))
    print("Cross validation accuracy for best learning rate = " + str(acc_dictionary[best_hp]))
    print("Total number of updates performed by the learning algorithm on training set = " + str(P.updates))
    print("Developement set accuracy = " + str(P_set[1]))
    print("Test set accuracy = " + str(P.accuracy))
    
    x_axis = list(range(1,21))
    y_axis = [x[1] for x in epoch_acc_dict]
    plt.plot(x_axis, y_axis)
    plt.ylim([1,100])
    
#     Uncomment the below line to see the plot
#    plt.show()

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

def aggresive_perceptron():
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
            
            p1.train(train1, lr, aggressive = True)
            p2.train(train2, lr, True)
            p3.train(train3, lr, True)
            p4.train(train4, lr, True)
            p5.train(train5, lr, True)
        
        p1.predict(test1)
        p2.predict(test2)
        p3.predict(test3)
        p4.predict(test4)
        p5.predict(test5)
        
        acc_dictionary[lr] = (p1.accuracy + p2.accuracy + p3.accuracy + p4.accuracy + p5.accuracy)/5
    
    best_hp = max(acc_dictionary, key=acc_dictionary.get)
    
    #Final Run
    train = de.extract(["Dataset/phishing.train"])
    dev = de.extract(["Dataset/phishing.dev"])
    test = de.extract(["Dataset/phishing.test"])
    
    epoch = 20
    
    epoch_acc_dict = []
    p = Perceptron()
        
    for i in range(epoch):
        np.random.shuffle(train)
        p.train(train, best_hp, True)
        p.predict(dev)
        epoch_acc_dict.append((copy.deepcopy(p), p.accuracy))
    
    P_set = max(epoch_acc_dict, key = lambda x:x[1])
    P = P_set[0]
    P.predict(test)
    
#   
#    print(P.accuracy)
    print("Best learning rate = " + str(best_hp))
    print("Cross validation accuracy for best learning rate = " + str(acc_dictionary[best_hp]))
    print("Total number of updates performed by the learning algorithm on training set = " + str(P.updates))
    print("Developement set accuracy = " + str(P_set[1]))
    print("Test set accuracy = " + str(P.accuracy))
    
    x_axis = list(range(1,21))
    y_axis = [x[1] for x in epoch_acc_dict]
    plt.plot(x_axis, y_axis)
    plt.ylim([1,100])
    
#     Uncomment the below line to see the plot
#    plt.show()
    
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
    
start = time.clock()

set_cross_validation()

#simple_perceptron()
#print()
#print()
#print()
#dynamic_perceptron()
#print()
#print()
#print()
#margin_perceptron()    
#print()
#print()
#print()    
#average_perceptron()

aggresive_perceptron()

timediff = time.clock() - start

print(timediff)
