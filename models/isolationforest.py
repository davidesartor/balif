from typing import Optional
from numpy.typing import NDArray
import numpy as np

from dataclasses import InitVar, dataclass, field

from .utils.binarytree import depth, left_child_idx, right_child_idx, expected_isolation_depth
from .utils.hyperplane import Hyperplane
from .utils import jitted_routines


@dataclass(eq=False)
class IsolationTree:
    """Isolation Tree is a binary tree for anomaly detection."""

    data: InitVar[NDArray[np.float64]]
    hyperplane_components: int | None = None
    seed: InitVar[Optional[int]] = None

    # tree properties
    psi: int = field(init=False)
    c_norm: float = field(init=False)
    n_features: int = field(init=False)
    max_depth: int = field(init=False)

    # node properties
    node_sizes: NDArray[np.int32] = field(init=False, repr=False)
    hyperplane_normals: NDArray[np.float64] = field(init=False, repr=False)
    hyperplane_intercepts: NDArray[np.float64] = field(init=False, repr=False)
    corrected_depths: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self, data: NDArray[np.float64], seed: Optional[int] = None):
        # initialize tree properties
        self.psi = data.shape[0]
        self.c_norm = expected_isolation_depth(self.psi)
        self.n_features = data.shape[1]
        self.max_depth = int(np.log2(self.psi))

        # initialize node properties
        n_nodes = 2 ** (self.max_depth + 1) - 1
        self.node_sizes = np.empty(n_nodes, dtype=np.int32)
        self.hyperplane_normals = np.empty((n_nodes, self.n_features), dtype=np.float64)
        self.hyperplane_intercepts = np.empty(n_nodes, dtype=np.float64)
        self.corrected_depths = np.empty(n_nodes, dtype=np.float64)

        # create the tree
        self.fit(data, seed=seed)

    def expand_tree_from(self, node_idx: int, data: NDArray[np.float64]) -> None:
        if depth(node_idx) > self.max_depth:
            return

        # create the node
        self.node_sizes[node_idx] = data.shape[0]
        self.corrected_depths[node_idx] = depth(node_idx)
        self.corrected_depths[node_idx] += expected_isolation_depth(self.node_sizes[node_idx])

        hyperplane = Hyperplane.random_splitting_plane(data, components=self.hyperplane_components)
        self.hyperplane_normals[node_idx] = hyperplane.normal
        self.hyperplane_intercepts[node_idx] = hyperplane.intercept

        # split the data and recurse
        distances = hyperplane.distance(data)
        self.expand_tree_from(left_child_idx(node_idx), data[distances <= 0])
        self.expand_tree_from(right_child_idx(node_idx), data[distances >= 0])

    def fit(self, data: NDArray[np.float64], seed: Optional[int] = None) -> None:
        """Create the isolation tree from the data."""
        if seed is not None:
            np.random.seed(seed)
        self.expand_tree_from(node_idx=0, data=data)

    def isolation_nodes_idxs(self, data: NDArray[np.float64]) -> NDArray[np.int32]:
        return jitted_routines.isolation_node_idxs(
            data, self.hyperplane_normals, self.hyperplane_intercepts, self.node_sizes
        )

    def predict(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Score the data points based on the node sizes along the paths."""
        correct_depths = self.corrected_depths[self.isolation_nodes_idxs(data)]
        scores = 2 ** (-correct_depths / self.c_norm)
        return np.array(scores)


@dataclass(eq=False)
class IsolationForest:
    """Isolation Forest is an unsupervised learning algorithm for anomaly detection."""

    n_trees: int = 128
    hyperplane_components: int | None = 1
    max_bagging_samples: int = 256

    trees: list[IsolationTree] = field(init=False, repr=False)

    @property
    def n_features(self) -> int:
        if len(self.trees) == 0:
            raise ValueError("model has not been fitted yet")
        return self.trees[0].n_features

    def create_tree(self, data: NDArray[np.float64]) -> IsolationTree:
        """Create a single estimator."""
        samplesize = min((self.max_bagging_samples, data.shape[0]))
        subsample = data[np.random.randint(data.shape[0], size=samplesize)]
        return IsolationTree(data=subsample, hyperplane_components=self.hyperplane_components)

    def fit(self, data: NDArray[np.float64], seed: Optional[int] = None):
        """Create the isolation forest ensemble from the data."""
        if seed is not None:
            np.random.seed(seed)
        self.trees = [self.create_tree(data) for _ in range(self.n_trees)]

    def predict(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict the anomaly score for each sample in the data."""
        # gather the anomaly scores according to individual estimators for all data points
        # single_tree_scores has shape: (n_trees, n_samples)
        single_tree_scores = [tree.predict(data) for tree in self.trees]

        # average the anomaly scores across all estimators using log2 to recover normalized depth
        ensamble_scores = 2 ** np.mean(np.log2(single_tree_scores), axis=0)
        return ensamble_scores

    def fit_predict(
        self, data: NDArray[np.float64], seed: Optional[int] = None
    ) -> NDArray[np.float64]:
        """Fit the model to the data and predict the anomaly score for each sample in the data."""
        self.fit(data, seed=seed)
        return self.predict(data)
