import numpy as np
import numpy.typing as npt
from typing import Optional, NamedTuple
from scipy.stats import beta
from pyod.models.base import BaseDetector
from pyod.models.iforest import IForest

DATA = npt.NDArray[np.float_]
SCORES = npt.NDArray[np.float_]
LABELS = npt.NDArray[np.int_]
REGIONS = npt.NDArray[np.int_]


class BetaDistr(NamedTuple):
    a: npt.NDArray[np.float_]
    b: npt.NDArray[np.float_]

    def mean(self):
        return self.a / (self.a + self.b)


class BayesianDetector(BaseDetector):
    @property
    def regions_score(self) -> SCORES:
        raise NotImplementedError  # shape:(n_estimators, max_regions)

    def estimators_apply(self, X: DATA) -> REGIONS:
        raise NotImplementedError  # shape:(n_estimators, n_samples)

    def __init__(
        self,
        *args,
        prior_sample_size=0.1,
        aggregation_strategy="geom",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prior_sample_size = prior_sample_size
        self.aggregation_strategy = aggregation_strategy

    def fit(self, X: DATA, y: Optional[LABELS] = None):
        super().fit(X, y)
        self.beliefs = self.unsupervised_beliefs(self.regions_score)
        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
        return self

    def unsupervised_beliefs(self, regions_score) -> BetaDistr:
        # fit a uniform prior, matching the contamination
        mean = self.contamination * np.ones(regions_score.shape)
        sample_size = self.prior_sample_size * np.ones(regions_score.shape)
        prior = BetaDistr(a=mean * sample_size, b=(1 - mean) * sample_size)

        # add positive obs matching the mean to detector scores
        regions_score = np.clip(regions_score, 0.01, 0.99)
        a_over_b = regions_score / (1 - regions_score)
        beliefs = BetaDistr(
            a=np.maximum(a_over_b * prior.b, prior.a),
            b=np.maximum(prior.a / a_over_b, prior.b),
        )
        assert np.allclose(beliefs.mean(), regions_score, equal_nan=True)
        return beliefs

    def decision_function(self, X: DATA) -> BetaDistr:
        regions = self.estimators_apply(X)  # shape: (n_estimators, n_samples)
        beliefs = BetaDistr(
            a=np.take_along_axis(self.beliefs.a, regions, axis=1),
            b=np.take_along_axis(self.beliefs.b, regions, axis=1),
        )  # shape: (n_estimators, n_samples)

        if self.aggregation_strategy == "geom":
            scores = np.exp(np.mean(np.log(beliefs.mean()), axis=0))
        return scores

    def update(self, X: DATA, y: LABELS, confidence=1.0):
        # TODO add support for sample-level confidence
        # TODO add support for multi-sample updates
        assert X.ndim == 1, "Update only supports single sample for now"
        X = X.view((1, -1))
        regions = self.estimators_apply(X)  # shape: (n_estimators, 1)

        # update a
        a_of_regions = np.take_along_axis(self.beliefs.a, regions, axis=1)
        a_of_regions += confidence * (y >= 1)
        np.put_along_axis(self.beliefs.a, regions, a_of_regions, axis=1)

        # update b
        b_of_regions = np.take_along_axis(self.beliefs.b, regions, axis=1)
        b_of_regions += confidence * (y == 0)
        np.put_along_axis(self.beliefs.b, regions, b_of_regions, axis=1)


class BAD_IForest(BayesianDetector, IForest):
    def estimators_apply(self, X: DATA) -> REGIONS:
        return np.stack([est.apply(X) for est in self.estimators_], axis=0)

    @property
    def regions_score(self) -> SCORES:
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
