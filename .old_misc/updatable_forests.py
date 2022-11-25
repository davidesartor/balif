from utils.forests import ExtendedTree, ExtendedIsolationForest, IsolationForest
import numpy as np
from utils.utils import BetaDistrib

def mode(alpha,beta):
    invalid = (alpha < 1) + (beta < 1)
    mode = (alpha-1)/(alpha+beta-2)
    mode[invalid] = np.where(alpha[invalid]>beta[invalid],0,1)
    return mode


def mean(alpha,beta):
    return (alpha)/(alpha+beta)


class BeliefTree():    
    def __init__(self, tree, pred="mean", force_prior=False):
        self.basetree = tree
        self.pred = pred
        prior_strat = force_prior or "balanced" if pred=="mean" else "bayes"
        self.set_priors(strategy=prior_strat)
    
    def set_priors(self, strategy="balanced"):
        offset = {"haldane":0.000001, "balanced":0.1, "jeffreys":0.5, "bayes":1}.get(strategy,1)
        h = self.basetree.corrected_depth
        c = self.basetree.correction_factor
        base_pred = 1/(1+c*np.power(2,-c*(-h+1)))
        size = offset/np.min((base_pred,1-base_pred),axis=0)
        if self.pred == "mean":
            self.alphas, self.betas = base_pred*size, (1-base_pred)*size
        if self.pred == "mode":
            self.alphas, self.betas = 1+base_pred*size, 1+(1-base_pred)*size
        
    def update_beliefs(self, x, is_anomaly):
        path, = np.array(self.basetree.apply(x[None,:]))
        self.alphas[path]+=is_anomaly
        self.betas[path]+=1-is_anomaly
            
    def get_beliefs_on(self, X):
        ids = self.basetree.leaf_ids(X)
        return self.alphas[ids],self.betas[ids]
    
    def predict(self, X, getparams=False, getdistr=False):
        alphas, betas = self.get_beliefs_on(X)
        if getparams: 
            return alphas, betas
        if getdistr: 
            return [BetaDistrib(a,b) for a,b in zip(alphas,betas)]
        if self.pred == "mean":
            return mean(alphas,betas)
        if self.pred == "mode":
            return mode(alphas,betas)
    
    
class BEIF():    
    def __init__(self, ensamble_prediction="naive", **forest_params):
        self.baseforest = ExtendedIsolationForest(**forest_params)
        self.ensamble_prediction = ensamble_prediction
        
    def fit(self, X, force_prior=False, seed=None):
        self.baseforest.fit(X, seed)
        pred = "mean" if self.ensamble_prediction == "naive" else "mode"
        self.trees = [BeliefTree(tree, pred=pred, force_prior=force_prior) for tree in self.baseforest.trees]
    
    def update(self, x, is_anomaly):
        for tree in self.trees:
            tree.update_beliefs(x, is_anomaly)
            
    def get_beliefs_on(self, X):
        """(x, alpha-beta, tree)"""
        return np.array([tree.get_beliefs_on(X) for tree in self.trees]).T
            
    def get_combined_belief_on(self, X):
        params = self.get_beliefs_on(X)
        alphas, betas = params[:,0,:], params[:,1,:]
        
        if self.ensamble_prediction == "likelihood":
            alphas, betas = 1+np.sum(alphas-1, axis=1), 1+np.sum(betas-1, axis=1)    
            
        elif self.ensamble_prediction == "mode":
            modes = np.mean(mode(alphas, betas),axis=1)
            concentrations = np.sum(alphas+betas-2, axis=1)         
            alphas, betas = 1+concentrations*modes, 1+concentrations*(1-modes)
            
        elif self.ensamble_prediction == "naive":
            means = np.mean(mean(alphas, betas),axis=1)
            samplesize = np.sum(alphas+betas, axis=1)      
            alphas, betas = samplesize*means, samplesize*(1-means)            

        return alphas, betas   
            
    def predict(self, X, getparams=False, getdistr=False):
        alphas, betas = self.get_combined_belief_on(X)
        if getparams: 
            return alphas, betas
        if getdistr: 
            return [BetaDistrib(a,b) for a,b in zip(alphas,betas)]
        
        if self.ensamble_prediction == "likelihood" or self.ensamble_prediction == "mode":
            return mode(alphas,betas)
        return mean(alphas,betas)
            
    
class BIF(BEIF):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseforest.cut_dimension = 1        