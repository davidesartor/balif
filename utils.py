import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

def plt_scatter_predictions(model, points=None, adaptive_range=True):
    grid = np.linspace(-0.9,0.9,50).astype(np.float64)
    heatmap = np.array([[model.predict(np.array([(x,y) for x in grid])) for y in grid]])[0]
    
    if adaptive_range:
        plt.imshow(heatmap,extent=(-0.9,0.9,-0.9,0.9), origin = "lower")  
    else:
        plt.imshow(heatmap,extent=(-0.9,0.9,-0.9,0.9), vmin=0, vmax=1, origin = "lower") 
    if points is not None:
        x,y = [p[0] for p in points],[p[1] for p in points]
        plt.plot(x,y,"rx")
        
        
def get_precision_recall(anom_scores,labels):
    precision = np.array([len(anom_scores[labels == 1][anom_scores[labels == 1]>=v])/len(anom_scores[anom_scores>=v]) for i,v in enumerate(sorted(anom_scores))])
    recall = np.array([len(anom_scores[labels == 1][anom_scores[labels == 1]>=v])/len(anom_scores[labels == 1]) for i,v in enumerate(sorted(anom_scores))])
    ap = np.trapz(precision[np.argsort(recall)], recall[np.argsort(recall)])
    return precision,recall, ap

def get_tpr_fpr(anom_scores,labels):
    tpr = np.array([len(anom_scores[labels == 1][anom_scores[labels == 1]>=v])/len(anom_scores[labels == 1]) for i,v in enumerate(sorted(anom_scores))])
    fpr = np.array([len(anom_scores[labels == 0][anom_scores[labels == 0]>=v])/len(anom_scores[labels == 0]) for i,v in enumerate(sorted(anom_scores))])
    auc = np.trapz(tpr[np.argsort(fpr)], fpr[np.argsort(fpr)])
    return tpr,fpr,auc


def get_accuracy(anom_scores,labels):
    precision,recall,ap = get_precision_recall(anom_scores,labels)
    tpr,fpr,auc = get_tpr_fpr(anom_scores,labels)
    return auc,ap


def evaluate_performance(model, data, labels, n_runs=1, max_iterations=50, size_query=0, test_size=0.2):
    performance_logs = {}    
    for run in tqdm(range(n_runs)):
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, stratify=labels)
        queried = np.zeros_like(y_train)
        
        model.fit(X_train)
        auc,ap = get_accuracy(model.predict(X_test),y_test)
        performance_logs[f"run n:{run}"] = [(auc,ap)]
        
        for i in range(max_iterations):   
            if not size_query or len(X_train)<size_query: eligible = np.arange(len(X_train))
            else: eligible = np.random.choice(np.arange(size_query),size_query)
                
            idx = eligible[model.ask_label(X_train[eligible],queried[eligible])]
            
            x,y = X_train[idx],y_train[idx]
            model.update(x,y)
            queried[idx] = True
            
            auc,ap = get_accuracy(model.predict(X_test),y_test)
            performance_logs[f"run n:{run}"].append((auc,ap))       
    return performance_logs


def plt_perf_evol(performance_logs, label="", plotroc=False, plotap=True):
    if plotroc:
        auc = np.array([[auc for auc,ap in perf_log] for _,perf_log in performance_logs.items()])
        auc_mean, auc_delta = get95percconf(auc)
        plt.plot(auc_mean,"o-",label="auc"+label)
        plt.fill_between(np.arange(auc.shape[1]),auc_mean-auc_delta,auc_mean+auc_delta,alpha=0.2)
    
    if plotap:
        ap = np.array([[ap for auc,ap in perf_log] for _,perf_log in performance_logs.items()])
        ap_mean, ap_delta = get95percconf(ap)
        plt.plot(ap_mean,"o-",label=label)
        plt.fill_between(np.arange(ap.shape[1]),ap_mean-ap_delta,ap_mean+ap_delta,alpha=0.2)
    plt.ylabel("ap")
    plt.xlabel("iteration")
    plt.legend()
    
    
def get95percconf(x, axis=0):
    k = x.shape[axis]
    t = {0: 0.0, 1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6:2.447, 7:2.365, 8:2.306, 9:2.262, 10:2.228}.get(k-1)
    mean = np.mean(x,axis=axis)
    delta = t*np.std(x,axis=axis)/np.sqrt(k)
    return mean, delta