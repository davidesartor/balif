import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utils.updatable_forests import BEIF, BIF
    
class BALEIF(BEIF):    
    def __init__(self, query_strategy="margin", **forest_params):
        super().__init__(**forest_params)
        self.querystrat = query_strategy
            
    def log_confidence_margins(self, X, threshold):
        predictions = self.predict(X, getdistr=True)
        return np.abs([distr.loglk_ratio(distr.mode(),threshold) for distr in predictions])
        
    def interest_on_info_for(self, X): 
        if self.querystrat == "random":
            importance = np.ones(len(X))
        if self.querystrat == "margin":
            importance = -self.log_confidence_margins(X,0.5)
        if self.querystrat == "boostedmargin":
            importance = -self.log_confidence_margins(X,0.75)
        if self.querystrat == "anomalous":
            importance = np.log(self.predict(X))
        return importance
    
    
class BALIF(BALEIF):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseforest.cut_dimension = 1 
        
    
class ALEIF(BALEIF):    
    def fit(self, X, seed=None):
        super().fit(X, seed=seed, force_prior="haldane")
        depthranges = [belieftree.basetree.depthrange for belieftree in self.trees]
        self.mindepth, self.maxdepth = np.min(depthranges), np.max(depthranges)
        
    def color_to_depth(self, colors):
        if self.ensamble_prediction == "piecewise":
            case_low = (2*colors*(1-self.maxdepth)+self.maxdepth)*(colors<=0.5)
            case_high = (2*colors*(self.mindepth-1)+2-self.mindepth)*(colors>=0.5)
            return case_low+case_high
        return -np.log2(colors)
        
    def get_combined_belief_on(self, X):
        params = self.get_beliefs_on(X)
        alphas, betas = params[:,0,:], params[:,1,:]
        node_colors = alphas/(alphas+betas)
        depths = self.color_to_depth(node_colors)
        
        means = np.power(2,-np.mean(depths,axis=1))
        samplesize = np.sum(alphas+betas, axis=1)      
        alphas, betas = samplesize*means, samplesize*(1-means)  
        return alphas, betas
    
    
class ALIF(ALEIF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseforest.cut_dimension = 1 
        
        
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
