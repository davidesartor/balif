from forests import ExtendedTree, ExtendedIsolationForest, IsolationForest
import numpy as np
from numba import njit
import copy

@njit
def mode(alpha,beta):
    return (alpha-1)/(alpha+beta-2)

class BeliefTree():    
    def __init__(self, tree):
        self.basetree = tree
        self.set_priors()
    
    def set_priors(self):
        weight = 0.1
        anomaly_scores = np.power(2,-self.basetree.corrected_depth)
        self.alphas = 1+weight*anomaly_scores
        self.betas = 1+weight*(1-anomaly_scores)
        
    def update_beliefs(self, x, is_anomaly):
        path, = np.array(self.basetree.apply(x[None,:]))
        self.alphas[path]+=is_anomaly
        self.betas[path]+=1-is_anomaly
            
    def get_beliefs_on(self, X):
        ids = self.basetree.leaf_ids(X)
        return self.alphas[ids],self.betas[ids]
    
    def predict(self, X):
        alphas, betas = self.get_beliefs_on(X)
        return mode(alphas,betas)
    
    def copy(self):
        return copy.deepcopy(self)
    
    
class BALEIF():    
    def __init__(self, isolation_forest_params = {}, query_strategy="margin", ensamble_prediction="naive"):
        self.baseforest = ExtendedIsolationForest(**isolation_forest_params)
        self.querystrat = query_strategy
        self.ensamble_prediction = ensamble_prediction
        
    def fit(self, X, locked_dims=None):
        self.baseforest.fit(X,locked_dims)
        self.trees = [BeliefTree(tree) for tree in self.baseforest.trees]
    
    def update(self, x, is_anomaly):
        for tree in self.trees:
            tree.update_beliefs(x, is_anomaly)
            
    def get_beliefs_on(self, X):
        """(x, alpha-beta, tree)"""
        return np.array([tree.get_beliefs_on(X) for tree in self.trees]).T
            
    def get_combined_belief_on(self, X):
        if self.ensamble_prediction == "maxlikelihood":
            alphas, betas = (1+np.sum(self.get_beliefs_on(X)-1, axis=2)).T        
            
        if self.ensamble_prediction == "naive":
            params = self.get_beliefs_on(X)
            alphas, betas = params[:,0,:], params[:,1,:]
            modes = np.mean(mode(alphas, betas),axis=1)
            concentrations = np.sum(alphas-1, axis=1) + np.sum(betas-1, axis=1)            
            alphas, betas = 1+concentrations*modes, 1+concentrations*(1-modes)
            
        return alphas, betas   
            
    def predict(self, X, distr = False):
        alphas, betas = self.get_combined_belief_on(X)       
        if not distr: 
            return mode(alphas,betas)
        else: 
            return [BetaDistrib(a,b) for a,b in zip(alphas, betas)]
            
    def confidence_margins(self, X, threshold):
        return [distr(distr.mode())/distr(threshold) for distr in self.predict(X, distr=True)]
        
    def interest_on_info_for(self, X): 
        if self.querystrat == "random":
            importance = np.ones(len(X))
        if self.querystrat == "margin":
            importance = -np.abs(np.log(self.confidence_margins(X,0.5)))
        if self.querystrat == "boostedmargin":
            importance = -np.abs(np.log(self.confidence_margins(X,0.8)))
        if self.querystrat == "anomalous":
            importance = np.log(self.predict(X))
            
        return importance
    
    def ask_label(self, data, queried):
        interest = self.interest_on_info_for(data)
        sample_prob = np.exp(interest)*(1-queried)
        sample_prob = sample_prob/np.sum(sample_prob)
        #idx = np.random.choice(np.arange(len(interest)),p=sample_prob)
        idx = np.argmax(sample_prob)
        return idx
    
class BALIF(BALEIF):    
    def __init__(self, isolation_forest_params = {}, query_strategy="margin", ensamble_prediction="naive"):
        self.baseforest = IsolationForest(**isolation_forest_params)
        self.querystrat = query_strategy
        self.ensamble_prediction = ensamble_prediction
    
    
class BetaDistrib():
    def __init__(self, a, b):
        self.a, self.b = a, b  
        
    def __call__(self,x):
        return x**(self.a-1)*(1-x)**(self.b-1) 

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