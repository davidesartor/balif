from dataclasses import InitVar, dataclass, field
from functools import cached_property
from typing import Optional, Self  # type: ignore
import numpy as np
from numpy.typing import NDArray

from joblib import Parallel, delayed

from models.jitted_routines import apply

from . import input_validation, binarytrees
from .utils import expected_isolation_depth, random_vector


@dataclass(frozen=True, eq=False, kw_only=True)
class IsolationTreeNode(binarytrees.BinaryTreeNode):
    data: NDArray[np.float64] = field(repr=False)
    hyperplane_components: int | None = None

    @property
    def size(self) -> int:
        return self.data.shape[0]

    @property
    def n_features(self) -> int:
        return self.hyperplane_normal.shape[0]

    @cached_property
    def corrected_depth(self) -> float:
        return self.depth + expected_isolation_depth(self.size)

    @cached_property
    def hyperplane_normal(self) -> NDArray[np.float64]:
        return random_vector(self.n_features, non_zero=self.hyperplane_components)

    @cached_property
    def hyperplane_intercept(self) -> np.float64:
        distances: NDArray[np.float64] = np.dot(
            self.data, self.hyperplane_normal)
        return np.random.uniform(min(distances), max(distances))

    @cached_property
    def children(self) -> tuple[Self, Self]:
        go_left, go_right, go_both = self.split_condition(self.data)
        left_child = type(self)(
            idx=binarytrees.left_child_idx(self.idx),
            data=self.data[go_left + go_both],
            hyperplane_components=self.hyperplane_components,
        )
        right_child = type(self)(
            idx=binarytrees.right_child_idx(self.idx),
            data=self.data[go_right + go_both],
            hyperplane_components=self.hyperplane_components,
        )
        return left_child, right_child

    def split_condition(
        self, data: NDArray[np.float64]
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
        distances = np.dot(data, self.hyperplane_normal) - \
            self.hyperplane_intercept
        return distances < 0, distances > 0, distances == 0


@dataclass(frozen=True, eq=False)
class IsolationTree:
    """Isolation Tree is a binary tree for anomaly detection."""
    data: InitVar[NDArray[np.float64]]
    hyperplane_components: int | None = None
    seed: InitVar[Optional[int]] = None

    psi: int = field(init=False)
    n_features: int = field(init=False)
    nodes: list[IsolationTreeNode] = field(
        init=False, repr=False, default_factory=list)

    @cached_property
    def c_norm(self) -> float:
        return expected_isolation_depth(self.psi)

    def __post_init__(self, data: NDArray[np.float64], seed: Optional[int] = None):
        object.__setattr__(self, "psi", data.shape[0])
        object.__setattr__(self, "n_features", data.shape[1])
        self.fit(data, seed=seed)

    def create_root_node(self, data: NDArray[np.float64]) -> IsolationTreeNode:
        """Initialize the root node of the tree."""
        return IsolationTreeNode(
            idx=0,
            data=data,
            hyperplane_components=self.hyperplane_components,
        )

    def expand_tree(self, root: IsolationTreeNode) -> list[IsolationTreeNode]:
        """Expand the tree from the root node until max depth is reached."""
        max_depth = int(np.log2(self.psi))
        nodes = [root]
        for node in nodes:
            if node.depth < max_depth:
                nodes.extend(node.children)
        return nodes

    def fit(self, data: NDArray[np.float64], seed: Optional[int] = None) -> None:
        """Create the isolation tree from the data."""
        if seed is not None:
            np.random.seed(seed)
        self.nodes.extend(self.expand_tree(root=self.create_root_node(data)))

    def apply(self, data: NDArray[np.float64]) -> NDArray[np.int32]:
        """Find the paths in the tree for each data point."""
        if data.ndim == 1:
            data = data[None, :]
        paths = []
        for datapoint in data:
            node = self.nodes[0]
            path = [node.idx]
            while not node.is_leaf:
                go_left, go_right, go_both = node.split_condition(datapoint)
                left_child, right_child = node.children
                if go_left.any():
                    node = left_child
                elif go_right.any():
                    node = right_child
                else:
                    node = left_child if np.random.rand() < 0.5 else right_child
                path.append(node.idx)
            paths.append(path)
        return np.array(paths)

    def isolation_node_idx(self, path: NDArray[np.int32]) -> np.int32:
        """find the isolation node in the path."""
        for idx in path:
            node = self.nodes[idx]
            if node.size <= 1:
                return idx
        return path[-1]

    def predict(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Score the data points based on the node sizes along the paths."""
        scores = []
        for path in self.apply(data):
            isolation_node = self.nodes[self.isolation_node_idx(path)]
            scores.append(2 ** (-isolation_node.corrected_depth / self.c_norm))
        return np.array(scores)


@dataclass(frozen=True, eq=False)
class IsolationForest:
    """Isolation Forest is an unsupervised learning algorithm for anomaly detection."""

    n_trees: int = 128
    hyperplane_components: int | None = 1
    max_bagging_samples: int = 256
    parallel_jobs: int = 1

    trees: list[IsolationTree] = field(
        init=False, repr=False, default_factory=list
    )

    @property
    def n_features(self) -> int:
        if len(self.trees) == 0:
            raise ValueError("model has not been fitted yet")
        return self.trees[0].n_features

    def create_tree(self, data: NDArray[np.float64]) -> IsolationTree:
        """Create a single estimator."""
        samplesize = min((self.max_bagging_samples, data.shape[0]))
        subsample = data[np.random.randint(data.shape[0], size=samplesize)]
        return IsolationTree(subsample, self.hyperplane_components)

    def fit(self, data: NDArray[np.float64], seed: Optional[int] = None):
        """Create the isolation forest ensemble from the data."""
        if seed is not None:
            np.random.seed(seed)

        if self.parallel_jobs == 1 or seed is not None:
            trees = [self.create_tree(data) for _ in range(self.n_trees)]
        else:
            trees = Parallel(n_jobs=self.parallel_jobs)(
                delayed(self.create_tree)(data) for _ in range(self.n_trees)
            )
        object.__setattr__(self, "trees", trees)

    def predict(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict the anomaly score for each sample in the data."""
        # gather the anomaly scores according to individual estimators for all data points
        # single_tree_scores has shape: (n_trees, n_samples)
        if self.parallel_jobs == 1:
            single_tree_scores = [tree.predict(data) for tree in self.trees]
        else:
            single_tree_scores = Parallel(n_jobs=self.parallel_jobs)(
                delayed(lambda model: model.predict(data))(tree) for tree in self.trees
            )
        # average the anomaly scores across all estimators using log2 to recover normalized depth
        # bit inneficient to exponentiate and then take the log2, but it's dominated by the tree traversal anyway
        # type: ignore
        ensamble_scores = 2 ** np.mean(np.log2(single_tree_scores), axis=0) # type: ignore
        return ensamble_scores

    def fit_predict(
        self, data: NDArray[np.float64], seed: Optional[int] = None
    ) -> NDArray[np.float64]:
        """Fit the model to the data and predict the anomaly score for each sample in the data."""
        self.fit(data, seed=seed)
        return self.predict(data)
