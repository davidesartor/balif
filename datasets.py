import numpy as np
import matplotlib.pyplot as plt
import scipy.io

datasets_names = [
    "annthyroid","breastw","cardio","ionosphere","letter",
    "mnist","optdigits","pendigits","pima","satellite",
    "satimage-2","thyroid","vertebral","vowels","wbc","wine",
]

def load_dataset(dataset_name=None, k=1):
    if dataset_name is None or dataset_name=="squarethoroid":
        data, labels = make_square_toroid(k=k)
    else:
        data, labels = load_data(dataset_name)
    return data.astype(np.float64), labels.astype(np.float64)
    

def load_data(name):
    mat = scipy.io.loadmat(f'datasets/{name}.mat')
    return mat["X"],mat["y"][:,0]

def make_square_toroid(verbose=False, k=1):
    std = 0.1
    central_cluster = np.random.uniform(-0.8,0.8,[int(1000*k),2])
    central_cluster = central_cluster[np.any(np.abs(central_cluster)>0.6,axis=1)]

    anomaly = np.random.uniform(-0.55,0.55,[int(50*k),2])
    data = np.vstack([central_cluster,anomaly])
    labels = np.hstack([np.zeros(central_cluster.shape[0]),np.ones(anomaly.shape[0])])
    
    if verbose:
        plt.figure(figsize=[5,5])
        plt.scatter(data[:,0],data[:,1],c=1-labels,cmap='Set1')
        #plt.scatter(anomaly[:,0],anomaly[:,1])
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        #plt.grid(True)
        plt.axis('off')
    
    return data,labels