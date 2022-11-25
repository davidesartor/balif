import numpy as np
from numba import njit, vectorize
from joblib import Parallel, delayed

@vectorize
def c_norm(k):
    if k <=1: return 0.0
    h_k = np.log(k-1)+0.57721
    return 2*h_k-2*(k-1)/k


@njit
def get_leaf_ids(X, child_left, child_right, normals, intercepts):
        res = np.zeros(len(X),dtype=np.int16)
        for i in range(len(X)):
            node_id = 0
            while child_left[node_id] or child_right[node_id]:
                if np.dot(X[i],normals[node_id]) <= intercepts[node_id]:
                    node_id = child_left[node_id]  
                else:
                    node_id = child_right[node_id]        
            res[i] = node_id
        return res
    

class ExtendedTree():
    def __init__(self, X, cut_dimension=None, random_state=None, normalization=None):
        self.random_seed = random_state
        self.fit(X, cut_dimension)
        
    def initialize_tree(self):
        self.child_left = np.zeros(2*self.psi, dtype=int)
        self.child_right = np.zeros(2*self.psi, dtype=int)
        self.normals = np.zeros((2*self.psi,self.d), dtype=float)
        self.intercepts = np.zeros(2*self.psi, dtype=float)
        self.node_size = np.zeros(2*self.psi, dtype=int)
        self.depth = np.zeros(2*self.psi, dtype=int)
        self.path_to = np.zeros(2*self.psi,dtype=object)
        
        self.node_count = 1
        self.path_to[0] = [0]
        
    def fit(self, X, cut_dimension=None):
        self.psi = len(X)
        self.d = X.shape[1]
        self.cut_dimension = cut_dimension or self.d
        self.max_depth = np.log2(self.psi)
        
        self.initialize_tree()
        self.extend_tree(node_id=0, X=X, depth=0)
        self.corrected_depth = (c_norm(self.node_size)+self.depth)/c_norm(self.psi)
        self.correction_factor = c_norm(self.psi)
        self.depthrange = (np.min(self.corrected_depth[:self.node_count]),np.max(self.corrected_depth[:self.node_count]))
        
    def create_new_node(self, parent_id):
        new_node_id = self.node_count
        self.node_count+=1
        self.path_to[new_node_id] = self.path_to[parent_id]+[new_node_id]
        self.depth[new_node_id] = self.depth[parent_id]+1     
        return new_node_id
    
    def sample_normal_vector(self):
        normals = np.zeros(self.d)
        selected_dims = np.random.choice(np.arange(self.d), size=self.cut_dimension, replace=False)
        normals[selected_dims] = np.random.randn(self.cut_dimension)
        return normals 
    
    def make_random_cut(self, node_id, X):
         # sample random normal vector
        self.normals[node_id] = self.sample_normal_vector()            
        # compute distances from plane (intercept in origin)
        dist = np.dot(X, self.normals[node_id])
        # sample intercept 
        self.intercepts[node_id] = np.random.uniform(np.min(dist),np.max(dist))
        # split condition for  X
        cond = dist <= self.intercepts[node_id]
        return X[cond==True], X[cond==False]
        
    def extend_tree(self, node_id, X, depth):
        self.node_size[node_id] = len(X)
        if depth >= self.max_depth or len(X)<=1:
            return
        
        X_left, X_right = self.make_random_cut(node_id, X)
        # add children 
        self.child_left[node_id] = self.create_new_node(node_id)
        self.child_right[node_id] = self.create_new_node(node_id)
        # recurse on children
        self.extend_tree(self.child_left[node_id], X_left, depth+1)        
        self.extend_tree(self.child_right[node_id], X_right, depth+1)
        
    def leaf_ids(self, X):
        return get_leaf_ids(X, self.child_left, self.child_right, self.normals, self.intercepts) 
                
    def apply(self, X):
        return self.path_to[self.leaf_ids(X)] 
    
    def predict(self, X):
        return self.corrected_depth[self.leaf_ids(X)]
    

class ExtendedIsolationForest():
    def __init__(self, n_estimators=100, max_samples="auto", cut_dimension=None):
        self.n_estimators = n_estimators
        self.max_samples = 256 if max_samples == "auto" else max_samples
        self.c_norm = c_norm(self.max_samples)
        self.cut_dimension = cut_dimension
        
    def fit(self, X, seed=None):
        samplesize = np.min((self.max_samples,len(X)))
        
        self.trees = [
            ExtendedTree(
                X[np.random.randint(len(X), size=samplesize)], 
                self.cut_dimension, 
                random_state=seed,
            ) 
            for _ in range(self.n_estimators)
        ]
        
    def predict(self, X):
        depths = Parallel(n_jobs=8)(
            delayed(lambda tree: tree.predict(X))(tree)
            for tree in self.trees
        )
        return np.power(2,-np.mean(depths, axis=0))
    
    
class IsolationForest(ExtendedIsolationForest):       
    def __init__(self, *args, **kwargs):
        super().__init__(*args, cut_dimension=1, **kwargs)
        


        
    
