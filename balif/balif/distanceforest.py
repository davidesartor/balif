import numpy as np
from numba import njit, vectorize
from utils.forests import ExtendedTree, ExtendedIsolationForest
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
                if np.linalg.norm(X[i]-normals[node_id]) <= intercepts[node_id]:
                    node_id = child_left[node_id]  
                else:
                    node_id = child_right[node_id]        
            res[i] = node_id
        return res

    
class DistanceTree(ExtendedTree):    
    def extend_tree(self, node_id, X, depth):
        self.node_size[node_id] = len(X)
        if depth >= self.max_depth or len(X)<=2:
            return
        
        X_left, X_right = self.make_random_cut(node_id, X)
        # add children 
        self.child_left[node_id] = self.create_new_node(node_id)
        self.child_right[node_id] = self.create_new_node(node_id)

        self.depth[self.child_left[node_id]] *= 2
        self.depth[self.child_right[node_id]] -= 0
        
        # recurse on children
        self.extend_tree(self.child_left[node_id], X_left, depth+1)        
        self.extend_tree(self.child_right[node_id], X_right, depth+1)
        
                
    def make_random_cut(self, node_id, X):
        # sample 3 random points from X
        points = X[np.random.choice(np.arange(len(X)), size=3, replace=False)]

        # compute paitwise distances
        dists = np.vstack([
            np.linalg.norm(points[0]-points[1]),
            np.linalg.norm(points[1]-points[2]),
            np.linalg.norm(points[2]-points[0]),
        ])

        # the radius is the minmum distance between two points
        self.intercepts[node_id] = np.min(dists)

        # the circle center is one of the two closest points
        self.normals[node_id] = points[np.argmin(dists)]
        
        # compute distances from the center of circle (normal)
        dist = np.linalg.norm(X-self.normals[node_id], axis=1)
        
        # split condition for  X
        cond = dist <= self.intercepts[node_id]
        return X[cond==True], X[cond==False]
        
    def leaf_ids(self, X):
        return get_leaf_ids(X, self.child_left, self.child_right, self.normals, self.intercepts) 
    
    def predict(self, X):
        return self.depth[self.leaf_ids(X)]

class DistanceIsolationForest(ExtendedIsolationForest):
    def __init__(self, n_estimators=100, max_samples="auto", cut_dimension=None):
        self.n_estimators = n_estimators
        self.max_samples = 256 if max_samples == "auto" else max_samples
        self.c_norm = c_norm(self.max_samples)
        self.cut_dimension = cut_dimension
        
    def fit(self, X, seed=None):
        samplesize = np.min((self.max_samples,len(X)))
        
        self.trees = [
            DistanceTree(
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
        return -np.log(np.mean(depths, axis=0))
    
class IsolationForest(ExtendedIsolationForest):       
    def __init__(self, *args, **kwargs):
        super().__init__(*args, cut_dimension=1, **kwargs)
        