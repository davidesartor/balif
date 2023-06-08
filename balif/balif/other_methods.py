import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utils.active_forests import BALEIF       
        
class RandomForest(BALEIF):
    def __init__(self, query_strategy="anomalous", ensamble_prediction="naive", **forest_params):
            self.forest_params = forest_params
            self.reset()
            self.querystrat = query_strategy
        
    def fit(self, X=None, seed=None):
        if seed: self.reset(seed)
        if self.trainX and self.trainy:
            self.baseforest.fit(np.array(self.trainX), np.array(self.trainy,dtype=int))
    
    def reset(self, seed=None):
        self.baseforest = RandomForestClassifier(**self.forest_params, random_state=seed)
        self.trainX, self.trainy = [],[]

    def update(self, x, is_anomaly):
        self.trainX.append(x)
        self.trainy.append(is_anomaly)
        self.fit()
            
    def predict(self, X):
        return self.baseforest.predict_proba(X)[:,-1] if self.trainX and self.trainy else np.zeros(len(X))