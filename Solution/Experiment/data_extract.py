import numpy as np

def extract(file_name):
    data = []
    labels = []
    
    with open(file_name) as f:
        for i in f.readlines():
            data.append(i.strip())
        
    for i,x in enumerate(data):
        if (x.split(' ')[0] == '+1'):
            labels.append(1)
        else:
            labels.append(-1)
        data[i] = x.split(' ')[1:]
    
    data = np.array(data)
    
    feat_count = []
    
    for x in data.flatten():
        feat_count.append(x.split(':')[0])
    
    feat_count = int(feat_count[-1])
    
    final_data = []
    
    for x in data:
        t = [0]*feat_count
        for y in x:
            p = y.split(':')
            t[int(p[0]) - 1] = float(p[1])
        final_data.append(t)
    
    final_data = np.array(final_data)
    return np.c_[final_data, labels]