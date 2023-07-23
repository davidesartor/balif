from dataclasses import dataclass, field
from typing import Optional
from joblib import Parallel, delayed
import numpy as np
from numpy.typing import NDArray

from .iforest import IsolationForest, IsolationTreeNode, IsolationTree

@dataclass(frozen=True, eq=False)
class AlifTreeNode(IsolationTreeNode):
    """Extention of a node of a binary isolation tree that can keep track of extra properties."""
    n_outliers: int = field(init=False, default=0)
    n_inliers: int = field(init=False, default=0)       

    def update(self, *args, new_ouliers: int = 0, new_inliers: int = 0) -> None:
        """Increment the counters of labelled data that reached this node."""
        object.__setattr__(self, "n_outliers", self.n_outliers + new_ouliers)
        object.__setattr__(self, "n_inliers", self.n_inliers + new_inliers)


class AlifTree(IsolationTree):
    """Modified version of the isolation tree for active learning."""
    
    def create_root_node(self, data: NDArray[np.float64]) -> IsolationTreeNode:
        """Initialize the root node of the tree."""
        return AlifTreeNode(0, data, self.hyperplane_components)
    
    def virtual_depth(self, node: AlifTreeNode) -> float:
        """Compute the virtual depth of a node."""
        if node.n_outliers + node.n_inliers == 0:
            return node.corrected_depth
        
        color = node.n_outliers / (node.n_outliers + node.n_inliers)
        if color <= 0.5: 
            max_depth = self.max_depth + self.c_norm
            return 2 * color * (self.c_norm - max_depth) + max_depth
        else: 
            min_depth = 1
            return 2 * color * (min_depth-self.c_norm) + 2*self.c_norm - min_depth

    def predict(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Score the data points based on the node sizes along the paths."""
        scores = []
        for path in self.apply(data):
            isolation_node = self.nodes[self.isolation_node_idx(path)]
            scores.append(2**(-self.virtual_depth(isolation_node)/self.c_norm))
        return np.array(scores)
    
    def update(self, *args, inlier_data: Optional[NDArray[np.float64]], outlier_data: Optional[NDArray[np.float64]]) -> None:
        """Update the model with new labelled data points."""
        inlier_counter = np.zeros(len(self.nodes), dtype=int)
        if inlier_data is not None:
            for path in self.apply(inlier_data):
                inlier_counter[path] += 1
                
        outlier_counter = np.zeros(len(self.nodes), dtype=int)
        if outlier_data is not None:
            for path in self.apply(outlier_data):
                outlier_counter[path] += 1
                
        for node, inlier_count, outlier_count in zip(self.nodes, inlier_counter, outlier_counter):
            node.update(new_inliers=inlier_count, new_ouliers=outlier_count)
            
    
class Alif(IsolationForest):
    """Active learning isolation forest."""

    def create_tree(self, data: NDArray[np.float64]) -> AlifTree:
        """Create a single estimator."""
        samplesize = min((self.max_bagging_samples, data.shape[0]))
        subsample = data[np.random.randint(data.shape[0], size=samplesize)]
        return AlifTree(subsample, self.hyperplane_components)
    
    def update(self, *args, inlier_data: Optional[NDArray[np.float64]] = None, outlier_data: Optional[NDArray[np.float64]] = None) -> None:
        """Update the model with new labelled data points."""
        for tree in self.trees: 
            tree.update(inlier_data=inlier_data, outlier_data=outlier_data)