from typing import Optional
from numpy.typing import NDArray
import numpy as np

from dataclasses import dataclass, field
from .utils.binarytree import expected_isolation_depth
from .isolationforest import IsolationForest, IsolationTree


@dataclass(eq=False)
class AlifTree(IsolationTree):
    """Modified version of the isolation tree for active learning."""

    # additional node properties
    n_outliers: NDArray[np.int32] = field(init=False, repr=False)
    n_inliers: NDArray[np.int32] = field(init=False, repr=False)
    virtual_depths: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self.n_outliers = np.zeros(len(self.node_sizes), dtype=np.int32)
        self.n_inliers = np.zeros(len(self.node_sizes), dtype=np.int32)
        self.virtual_depths = self.corrected_depths.copy()

    def labels2virtualdepth(
        self, n_outliers: NDArray[np.int32], n_inliers: NDArray[np.int32]
    ) -> NDArray[np.float64]:
        """Compute the virtual depth of a node."""
        color = n_outliers / (n_outliers + n_inliers)
        min_depth, max_depth = 1, self.max_depth + self.c_norm
        virtual_depths = (2 * color * (self.c_norm - max_depth) + max_depth) * (color <= 0.5) + (
            2 * color * (min_depth - self.c_norm) + 2 * self.c_norm - min_depth
        ) * (color >= 0.5)
        return virtual_depths

    def predict(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Score the data points based on the node sizes along the paths."""
        virtual_depths = self.virtual_depths[self.isolation_nodes_idxs(data)]
        scores = 2 ** (-virtual_depths / expected_isolation_depth(self.psi))
        return np.array(scores)

    def update(
        self,
        *args,
        inlier_data: Optional[NDArray[np.float64]],
        outlier_data: Optional[NDArray[np.float64]],
    ) -> None:
        """Update the model with new labelled data points."""
        if inlier_data is not None:
            if inlier_data.ndim == 1:
                inlier_data = inlier_data[None, :]
            terminal_nodes = self.isolation_nodes_idxs(inlier_data)
            self.n_inliers[terminal_nodes] += 1
            self.virtual_depths[terminal_nodes] = self.labels2virtualdepth(
                self.n_outliers[terminal_nodes], self.n_inliers[terminal_nodes]
            )
        if outlier_data is not None:
            if outlier_data.ndim == 1:
                outlier_data = outlier_data[None, :]
            terminal_nodes = self.isolation_nodes_idxs(outlier_data)
            self.n_outliers[terminal_nodes] += 1
            self.virtual_depths[terminal_nodes] = self.labels2virtualdepth(
                self.n_outliers[terminal_nodes], self.n_inliers[terminal_nodes]
            )


@dataclass(eq=False)
class Alif(IsolationForest):
    """Active learning isolation forest."""

    trees: list[AlifTree] = field(init=False, repr=False, default_factory=list)

    def create_tree(self, data: NDArray[np.float64]) -> AlifTree:
        """Create a single estimator."""
        samplesize = min((self.max_bagging_samples, data.shape[0]))
        subsample = data[np.random.randint(data.shape[0], size=samplesize)]
        return AlifTree(data=subsample, hyperplane_components=self.hyperplane_components)

    def update(
        self,
        *,
        inlier_data: Optional[NDArray[np.float64]] = None,
        outlier_data: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Update the model with new labelled data points."""
        for tree in self.trees:
            tree.update(inlier_data=inlier_data, outlier_data=outlier_data)

    def interest(self, data: NDArray[np.float64], strategy: str = "score"):
        """Compute the interest of the data points."""
        if strategy == "score":
            return self.predict(data)
        elif strategy == "random":
            return np.random.rand(data.shape[0])
        else:
            raise ValueError(f"Unknown interest strategy: {strategy}")
