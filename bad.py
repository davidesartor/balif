from typing import Literal, Optional, NamedTuple
from jaxtyping import Float, Int, Shaped

import numpy as np
from pyod.models.base import BaseDetector


class BayesianDetector(BaseDetector):
    @property
    def regions_score(self) -> Float[np.ndarray, "estimators regions"]:
        raise NotImplementedError

    def estimators_apply(
        self, X: Float[np.ndarray, "samples features"]
    ) -> Int[np.ndarray, "samples estimators"]:
        raise NotImplementedError

    def __init__(
        self,
        *args,
        prior_sample_size=0.1,
        aggregation_method="arithmetic_mean",
        reprocess_decision_scores=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prior_sample_size = prior_sample_size
        self.aggregation_method = aggregation_method
        self.reprocess_decision_scores = reprocess_decision_scores

    def fit(
        self,
        X: Float[np.ndarray, "samples features"],
        y: Optional[Int[np.ndarray, "samples 1"]] = None,
    ):
        super().fit(X, y)
        self.beliefs = EnsembleBeliefs.from_scores(
            regions_score=self.regions_score,
            contamination=self.contamination,
            prior_sample_size=self.prior_sample_size,
        )
        if self.reprocess_decision_scores:
            self.decision_scores_ = self.decision_function(X)
            self._process_decision_scores()
        return self

    def decision_function(
        self, X: Float[np.ndarray, "samples features"]
    ) -> Float[np.ndarray, "samples"]:
        regions = self.estimators_apply(X)
        scores = self.beliefs.aggregate(regions, self.aggregation_method)
        return scores

    def update(
        self,
        X: Float[np.ndarray, "samples features"],
        y: Int[np.ndarray, "samples 1"],
        confidence: float | Float[np.ndarray, "#samples"] = 1.0,
    ):
        regions = self.estimators_apply(X)
        da = confidence * (y >= 1).flatten()
        db = confidence * (y == 0).flatten()
        self.beliefs.update(regions, da, db)


class BetaDistr(NamedTuple):
    a: Float[np.ndarray, "..."]
    b: Float[np.ndarray, "..."]

    def mean(self):
        return self.a / (self.a + self.b)


class EnsembleBeliefs(BetaDistr):
    a: Float[np.ndarray, "estimators regions"]
    b: Float[np.ndarray, "estimators regions"]

    @classmethod
    def from_scores(
        cls,
        regions_score: Float[np.ndarray, "estimators regions"],
        contamination: float = 0.1,
        prior_sample_size: float = 0.1,
    ):
        # flat prior, matching the contamination
        prior_a = contamination * prior_sample_size
        prior_b = (1 - contamination) * prior_sample_size

        # add positive obs matching the mean to detector scores
        regions_score = np.clip(regions_score, 0.01, 0.99)
        a_over_b = regions_score / (1 - regions_score)
        a = np.maximum(a_over_b * prior_b, prior_a)
        b = np.maximum(prior_a / a_over_b, prior_b)
        return cls(a=a, b=b)

    def update(
        self,
        samples_regions: Int[np.ndarray, "samples estimators"],
        da: Float[np.ndarray, "samples"],
        db: Float[np.ndarray, "samples"],
    ):
        a, b = self.gather(samples_regions)
        a = a + da[:, None]
        b = b + db[:, None]
        np.put_along_axis(self.a.T, samples_regions, a, axis=0)
        np.put_along_axis(self.b.T, samples_regions, b, axis=0)

    def gather(
        self, samples_regions: Int[np.ndarray, "samples estimators"]
    ) -> Shaped[BetaDistr, "samples estimators"]:
        a = np.take_along_axis(self.a.T, samples_regions, axis=0)
        b = np.take_along_axis(self.b.T, samples_regions, axis=0)
        return BetaDistr(a=a, b=b)

    def aggregate(
        self, samples_regions: Int[np.ndarray, "samples estimators"], method: str
    ) -> Float[np.ndarray, "samples"]:
        beliefs = self.gather(samples_regions)
        if method == "arithmetic_mean":
            return np.mean(beliefs.mean(), axis=-1)
        elif method == "geometric_mean":
            return np.exp(np.mean(np.log(beliefs.mean()), axis=-1))
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
