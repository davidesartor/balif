from jaxtyping import Float, Int
import numpy as np
from pyod.models.iforest import IForest
from sklearn.ensemble._iforest import _average_path_length  # type: ignore
from .bad import BayesianDetector


class BADIForest(BayesianDetector, IForest):
    def estimators_apply(
        self, X: Float[np.ndarray, "samples features"]
    ) -> Int[np.ndarray, "estimators samples"]:
        regions = [est.apply(X, check_input=False) for est in self.estimators_]
        return np.stack(regions, axis=0)

    @property
    def regions_score(self) -> Float[np.ndarray, "estimators regions"]:
        max_len = max(est.tree_.node_count for est in self.estimators_)
        scores = [tree_regions_score(est.tree_, rescale=True) for est in self.estimators_]
        padded_scores = [np.pad(s, (0, max_len - len(s)), constant_values=np.nan) for s in scores]
        return np.stack(padded_scores, axis=0)


def tree_regions_score(tree, rescale=True):
    depths = tree.compute_node_depths() - 1
    extra_depth = _average_path_length(tree.n_node_samples)
    normalization = extra_depth[0]
    scores = np.power(2, -(depths + extra_depth) / normalization)
    if rescale:
        min_score = 2 ** (-1 - tree.max_depth / normalization)
        max_score = 2 ** (-1 / normalization)
        scores = (scores - min_score) / (max_score - min_score)
    return scores
