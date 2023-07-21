from __future__ import annotations
from dataclasses import InitVar, dataclass, field
from functools import cached_property
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from joblib import Parallel, delayed

from .input_validation import validate_dtype, validate_array_dimension, validate_array_shape
from .utils import expected_isolation_depth, random_vector


@dataclass(frozen=True)
class BinaryTreeNode:
    idx: int

    @cached_property
    def depth(self) -> int:
        return int(np.log2(self.idx + 1))

    @property
    def left_child_idx(self) -> int:
        return 2 * self.idx + 1

    @property
    def right_child_idx(self) -> int:
        return 2 * self.idx + 2

    @cached_property
    def path_to(self) -> NDArray[np.int32]:
        """Indices of nodes on the path from the root to the current node (both included)."""
        path = [node_idx := self.idx]
        while node_idx > 0:
            node_idx = (node_idx - 1) // 2
            path.append(node_idx)
        return np.array(path[::-1], dtype=int)


@dataclass(frozen=True, eq=False)
class IsolationTreeNode(BinaryTreeNode):
    data: NDArray[np.float64] = field(repr=False)
    hyperplane_components: int | None = None

    @property
    def size(self) -> int:
        return self.data.shape[0]

    @property
    def n_features(self) -> int:
        return self.data.shape[1]

    @cached_property
    def corrected_depth(self) -> float:
        return self.depth + expected_isolation_depth(self.size)

    @cached_property
    def hyperplane_normal(self) -> NDArray[np.float64]:
        return random_vector(self.n_features, non_zero=self.hyperplane_components)

    @cached_property
    def hyperplane_intercept(self) -> np.float64:
        distances: NDArray[np.float64] = np.dot(self.data, self.hyperplane_normal)
        return np.random.uniform(min(distances), max(distances))

    def distance_from_hyperplane(
        self, data: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.dot(data, self.hyperplane_normal) - self.hyperplane_intercept

    def split_condition(
        self, data: NDArray[np.float64]
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
        distances = self.distance_from_hyperplane(data)
        return distances < 0, distances > 0, distances == 0

    def split(self) -> tuple[Self, Self]:  # type: ignore
        go_left, go_right, go_both = self.split_condition(self.data)
        left_child = type(self)(
            self.left_child_idx,
            self.data[go_left + go_both],
            self.hyperplane_components,
        )
        right_child = type(self)(
            self.right_child_idx,
            self.data[go_right + go_both],
            self.hyperplane_components,
        )
        return left_child, right_child


@dataclass(frozen=True, eq=False)
class IsolationTree:
    data: NDArray[np.float64] = field(repr=False)
    hyperplane_components: int | None = field(default=1, repr=False)
    seed: InitVar[Optional[int]] = None

    nodes: NDArray[IsolationTreeNode] = field(init=False, repr=False)  # type: ignore

    @cached_property
    def max_depth(self) -> int:
        return int(np.log2(self.data.shape[0]))

    @cached_property
    def c_norm(self) -> float:
        return expected_isolation_depth(self.nodes[0].size)

    @property
    def n_features(self) -> int:
        return self.nodes[0].n_features

    def __post_init__(self, seed: Optional[int] = None) -> None:
        self.fit(self.data, seed)

    def create_root_node(self, data: NDArray[np.float64]) -> IsolationTreeNode:
        """Initialize the root node of the tree."""
        return IsolationTreeNode(0, data, self.hyperplane_components)

    def expand_tree(self, root: IsolationTreeNode) -> list[IsolationTreeNode]:
        """Expand the tree from the root node until log2(len(data)) depth is reached."""
        nodes = [root]
        for node in nodes:
            if node.depth + 1 <= self.max_depth:
                nodes.extend(node.split())
        return nodes

    def fit(self, data: NDArray[np.float64], seed: Optional[int] = None) -> None:
        """Create the isolation tree from the data."""
        if seed is not None:
            np.random.seed(seed)
        nodes = self.expand_tree(root=self.create_root_node(data))
        object.__setattr__(self, "nodes", np.array(nodes, dtype=object))

    def leaf_node(self, data_point: NDArray[np.float64]) -> IsolationTreeNode:
        """Find the leaf node that contains the data point."""        
        node: IsolationTreeNode = self.nodes[0]
        while node.depth < self.max_depth:
            go_left, go_right, go_both = node.split_condition(data_point)
            if go_left or go_both:
                node = self.nodes[node.left_child_idx]
            elif go_right:
                node = self.nodes[node.right_child_idx]
        return node

    def apply(self, data: NDArray[np.float64]) -> NDArray[np.int32]:
        """Find the paths in the tree for each data point."""
        if data.ndim == 1:
            data = data[None, :]
        paths = np.array([self.leaf_node(x).path_to for x in data])
        return paths

    def predict(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Score the data points based on the node sizes along the paths."""
        scores = []
        for path in self.nodes[self.apply(data)]:
            truncated_path = [node for node in path if node.size > 1]
            isolation_node = path[min(len(truncated_path), len(path) - 1)]
            scores.append(2 ** (-isolation_node.corrected_depth / self.c_norm))
        return np.array(scores)


@dataclass(frozen=True, eq=False)
class IsolationForest:
    """Isolation Forest is an unsupervised learning algorithm for anomaly detection."""

    n_trees: int = 128
    hyperplane_components: int | None = 1
    max_bagging_samples: int = 256
    parallel_jobs: int = 1

    trees: NDArray[IsolationTree] = field(init=False, repr=False)  # type: ignore

    @property
    def n_features(self) -> int:
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

        data = validate_dtype(field_name="data", value=data, expected_dtype=np.float64)
        validate_array_dimension(field_name="data", value=data, expected_ndim=(1,2))
        
        if self.parallel_jobs == 1 or seed is not None:
            trees = [self.create_tree(data) for _ in range(self.n_trees)]
        else:
            trees = Parallel(n_jobs=self.parallel_jobs)(
                delayed(self.create_tree)(data) for _ in range(self.n_trees)
            )
        object.__setattr__(self, "trees", np.array(trees, dtype=object))

    def predict(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict the anomaly score for each sample in the data."""
        data = validate_dtype(field_name="data", value=data, expected_dtype=np.float64)
        validate_array_dimension(field_name="data", value=data, expected_ndim=(1,2))
        validate_array_shape(field_name="data", value=data, expected_shape=(..., self.n_features))

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
        ensamble_scores = 2 ** np.mean(np.log2(single_tree_scores), axis=0)  # type: ignore
        return ensamble_scores

    def fit_predict(
        self, data: NDArray[np.float64], seed: Optional[int] = None
    ) -> NDArray[np.float64]:
        """Fit the model to the data and predict the anomaly score for each sample in the data."""
        self.fit(data, seed=seed)
        return self.predict(data)
