import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
    

def plt_scatter_predictions(model, points=None, adaptive_range=True, resolution=50):
    grid = np.linspace(-0.9,0.9,resolution).astype(np.float64)
    heatmap = np.array([[model.predict(np.array([(x,y) for x in grid])) for y in grid]])[0]
    
    if adaptive_range:
        plt.imshow(heatmap,extent=(-0.9,0.9,-0.9,0.9), origin = "lower", cmap="coolwarm") 
        plt.colorbar()
    else:
        plt.imshow(heatmap,extent=(-0.9,0.9,-0.9,0.9), vmin=0, vmax=1, origin = "lower", cmap="coolwarm") 
    
    if points is not None:
        x,y,l = points[:,0], points[:,1], points[:,2]
        plt.plot(x[l==1],y[l==1],"o", c="gold")
        plt.plot(x[l==0],y[l==0],"o", c="midnightblue")
        
        
def get_precision_recall(anom_scores,labels):
    scores_anomalies = anom_scores[labels==1]
    anomalies = len(labels[labels==1])
    positives = np.array([len(anom_scores[anom_scores>=v]) for v in np.linspace(0,1,100)])
    true_positives = np.array([len(scores_anomalies[scores_anomalies>=v]) for v in np.linspace(0,1,100)])

    precision = true_positives/positives
    recall = true_positives/anomalies
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


def evaluate_performance(model, data, labels, n_runs=1, max_iterations=25, size_query=0, test_size=0.5, jobs=4):
    performance_logs = {}  
    runs_results = Parallel(n_jobs=jobs)(
        delayed(perform_run)(run, model, data, labels, max_iterations, size_query, test_size) 
        for run in range(n_runs)
    )  
    for run in range(n_runs):
        performance_logs[f"run n:{run}"] = runs_results[run]
    return performance_logs

def perform_run(run, model, data, labels, max_iterations, size_query, test_size):
    np.random.seed(run)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, stratify=labels)
    queried_points = []
    
    model.fit(X_train, seed=run)
    prediction = model.predict(X_test)        
    results = [(roc_auc_score(y_test,prediction),average_precision_score(y_test,prediction))]
    
    for i in range(max_iterations):   
        if not size_query or len(X_train)<size_query: eligible = np.arange(len(X_train))
        else: eligible = np.random.choice(np.arange(size_query),size_query)
            
        interest = model.interest_on_info_for(X_train[eligible])
        selected = eligible[np.argmax(interest)]

        model.update(X_train[selected], y_train[selected])
        queried_points.append((*X_train[selected], y_train[selected]))
        X_train = X_train[np.arange(len(X_train)) != selected]
        y_train = y_train[np.arange(len(y_train)) != selected]
        
        prediction = model.predict(X_test)
        results.append((roc_auc_score(y_test,prediction),average_precision_score(y_test,prediction)))        
    return results


def plt_perf_evol(performance_logs, label=None, style="o-", color=None, plotroc=False, plotap=True):
    if plotroc:
        auc = np.array([[auc for auc,ap in perf_log] for _,perf_log in performance_logs.items()])
        auc_mean, auc_delta = get95percconf(auc)
        plt.plot(auc_mean,"o-",label="auc"+label)
        plt.fill_between(np.arange(auc.shape[1]),auc_mean-auc_delta,auc_mean+auc_delta,alpha=0.2)
    
    if plotap:
        ap = np.array([[ap for auc,ap in perf_log] for _,perf_log in performance_logs.items()])
        ap_mean, ap_delta = get95percconf(ap)
        if color is None: 
            plt.plot(ap_mean,style,label=label)
            plt.fill_between(np.arange(ap.shape[1]),ap_mean-ap_delta,ap_mean+ap_delta,alpha=0.2)
        else: 
            plt.plot(ap_mean,style,c=color,label=label)
            plt.fill_between(np.arange(ap.shape[1]),ap_mean-ap_delta,ap_mean+ap_delta,alpha=0.2, color=color)
        
    plt.ylabel("ap")
    plt.xlabel("labelled points")
    if label is not None: plt.legend()
    

from scipy.stats import t
def get95percconf(x, axis=0):
    k = x.shape[axis]
    ts = t.ppf(1-0.025, k-1)
    mean = np.mean(x,axis=axis)
    delta = ts*np.std(x,axis=axis,ddof=1)/np.sqrt(k)
    return mean, delta


class BetaDistrib():
    def __init__(self, a, b):
        self.a, self.b = a, b  
        
    def lk_ratio(self, x1, x2):
        return self(x1)/self(x2)
    
    def loglk_ratio(self, x1, x2):
        return np.log(self(x1))-np.log(self(x2))
        
    def __call__(self, x):
        return beta.pdf(x, self.a, self.b)

    def update(self, da=0, db=0):
        self.a += da
        self.b += db
        
    def expectation(self):
        return self.a/(self.a+self.b)
    
    def mean(self):
        return self.expectation()
    
    def mode(self):
        return (self.a-1)/(self.a+self.b-2)     
    
    def variance(self):
        return (self.a*self.b)/(self.a+self.b+1)/(self.a+self.b)**2
    
    def std(self):
        return np.sqrt(self.variance())
    
    def mean_abs_dev(self):
        return np.sqrt(2/np.pi)*(12+7/(self.a+self.b) -1/self.a -1/self.b)/12
    
    def __repr__(self):
        return f"BetaDistr(α={self.a}, β={self.b})"