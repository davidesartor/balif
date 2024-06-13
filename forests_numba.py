from dataclasses import dataclass, field
import numpy.typing as npt
import numpy as np
import numba


@numba.experimental.jitclass()
class IsolationTree:
    max_depth: int
    reached: numba.uint16[:]
    normals: numba.float64[:, :]
    intercepts: numba.float64[:]
    normalized_depth: numba.float64[:]

    def __init__(self, data: npt.NDArray[np.float64], hyperplane_components: int):
        samples, dim = data.shape
        self.max_depth = int(np.ceil(np.log2(samples)))
        nodes, leaves = 2**self.max_depth - 1, 2**self.max_depth

        self.normals = np.empty((nodes, dim), dtype=np.float64)
        self.intercepts = np.empty((nodes,), dtype=np.float64)

        reached = np.zeros((nodes + leaves, samples), dtype="bool")
        reached[0, :] = True
        for idx in range(nodes):
            normal, intercept, distances = self.sample_hyperplane_split(
                data[reached[idx]], hyperplane_components
            )
            self.normals[idx] = normal
            self.intercepts[idx] = intercept
            reached[2 * idx + 1, reached[idx]] = distances >= 0
            reached[2 * idx + 2, reached[idx]] = distances <= 0
        self.reached = reached.sum(axis=-1, dtype=np.uint16)

        depth = np.floor(np.log2(np.arange(nodes + leaves) + 1))
        expected = self.expected_depth(self.reached)
        self.normalized_depth = (depth + expected) / expected[0]

    @staticmethod
    def expected_depth(node_size: npt.NDArray[np.uint16]) -> npt.NDArray[np.float64]:
        """Compute the expected isolation depth for a node with n data points."""
        EULER_MASCHERONI = 0.5772156649
        harmonic_number = np.log(node_size - 1) + EULER_MASCHERONI
        expected_depth = 2 * harmonic_number - 2 * (node_size - 1) / node_size
        return np.where(node_size > 1, expected_depth, 0.0).astype(np.float64)

    @staticmethod
    def sample_hyperplane_split(
        data: npt.NDArray[np.float64],
        components: int,
    ):
        samples, dim = data.shape
        idx = np.random.choice(dim, components, replace=False)
        values = np.random.normal(0.0, 1.0, components).astype(np.float64)
        normal = np.zeros(dim, dtype=np.float64)
        normal[idx] = values

        distances = np.dot(data, normal)
        intercept = np.random.uniform(distances.min(), distances.max())
        distances -= intercept
        return normal, intercept, distances

    def path(self, data: npt.NDArray[np.float64]):
        samples, dim = data.shape
        (nodes,) = self.reached.shape
        path_lenght = int(np.log2(nodes + 1))

        paths = np.zeros((samples, path_lenght), dtype=np.int16)
        for depth in range(path_lenght):
            current_nodes = paths[:, depth]

            normals = self.normals[current_nodes]
            intercepts = self.intercepts[current_nodes]
            distances = (data * normals).sum(-1) - intercepts

            childs = 2 * current_nodes + 1 + (distances <= 0)
            paths[:, depth + 1] = np.where(
                self.reached[current_nodes] > 1, childs, current_nodes
            )
        return paths

    def score(self, data: npt.NDArray[np.float64]):
        samples, dim = data.shape
        isolation_node = self.path(data)[:, -1]
        return 2 ** -self.normalized_depth[isolation_node]


@dataclass
class IsolationForest:
    n_estimators: int = 128
    max_samples: int = 256
    bootstrap: bool = True
    standardize: bool = False
    hyperplane_components: int = 1
    # p_normal_idx: Literal["uniform", "range", "covariant"] = "uniform"
    # p_normal_value: Literal["uniform", "range", "covariant"] = "uniform"
    # p_intercept: Literal["uniform", "normal"] = "uniform"

    trees: list[IsolationTree] = field(init=False)

    def fit(self, data: npt.NDArray[np.float64]):
        samples, dim = data.shape
        max_samples = min(self.max_samples, samples)
        hyperplane_components = self.hyperplane_components
        if hyperplane_components <= 0:
            hyperplane_components = dim

        # if self.standardize:
        #     mean = np.mean(data, axis=0)
        #     std = np.std(data, axis=0)
        #     data = (data - mean) / np.where(std > 0, std, 1.0)

        subsamples_idx = [
            np.random.choice(samples, max_samples, replace=self.bootstrap)
            for _ in range(self.n_estimators)
        ]

        self.trees = [
            IsolationTree(data[subsample], hyperplane_components)
            for subsample in subsamples_idx
        ]

    def score(self, data: npt.NDArray[np.float64]):
        tree_scores = np.array([tree.score(data) for tree in self.trees])
        return np.exp(np.mean(np.log(tree_scores), axis=0))
