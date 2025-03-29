from jaxtyping import Float, Int
import numpy as np
from pyod.models.iforest import IForest
from bad import BayesianDetector


class BAD_IForest(BayesianDetector, IForest):
    def estimators_apply(
        self, X: Float[np.ndarray, "samples features"]
    ) -> Int[np.ndarray, "estimators samples"]:
        regions = [est.apply(X, check_input=False) for est in self.estimators_]
        return np.stack(regions, axis=0)

    @property
    def regions_score(self) -> Float[np.ndarray, "estimators regions"]:
        max_len = max(est.tree_.node_count for est in self.estimators_)
        scores = [
            tree_regions_score(est.tree_, rescale_scores=True, pad_len=max_len)
            for est in self.estimators_
        ]
        return np.stack(scores, axis=0)


def tree_regions_score(tree, rescale_scores=True, pad_len=None):
    depths = tree.compute_node_depths() - 1
    extra_depth = avg_path_BST(tree.n_node_samples)
    normalization = avg_path_BST(tree.n_node_samples[0])
    scores = np.power(2, -(depths + extra_depth) / normalization)
    if rescale_scores:
        min_score = 2 ** (-1 - tree.max_depth / normalization)
        max_score = 2 ** (-1 / normalization)
        scores = (scores - min_score) / (max_score - min_score)
    if pad_len is not None:
        scores = np.pad(scores, (0, pad_len - len(scores)), constant_values=np.nan)
    return scores


@np.vectorize
def avg_path_BST(n: int) -> float:
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2 * (np.log(n - 1) + np.euler_gamma - (n - 1) / n)
