from typing import Literal, Optional, NamedTuple
from jaxtyping import Float, Int, Shaped

import numpy as np
from scipy.stats import beta
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
        prior_sample_size: float = 0.1,
        aggregation_method: Literal["sum", "moment"] = "sum",
        interest_method: Literal["margin", "anom"] = "margin",
        reprocess_decision_scores: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prior_sample_size = prior_sample_size
        self.aggregation_method: Literal["sum", "moment"] = aggregation_method
        self.interest_method: Literal["margin", "anom"] = interest_method
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
        beliefs = self.beliefs.aggregate(regions, self.aggregation_method)
        scores = beliefs.mean()
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

    def interest(self, X: Float[np.ndarray, "samples features"]) -> Float[np.ndarray, "samples"]:
        regions = self.estimators_apply(X)
        beliefs = self.beliefs.aggregate(regions, method=self.aggregation_method)
        if self.interest_method == "margin":
            return np.exp(-beliefs.log_margin())
        elif self.interest_method == "anom":
            return beliefs.mean()
        else:
            raise ValueError(f"Unknown interest method: {self.interest_method}")


class BetaDistr(NamedTuple):
    a: Float[np.ndarray, "..."]
    b: Float[np.ndarray, "..."]

    def mean(self):
        return self.a / (self.a + self.b)

    def mode(self):
        return np.where(
            np.minimum(self.a, self.b) > 1,
            (self.a - 1) / (self.a + self.b - 2),
            (self.a > self.b).astype(float),
        )

    def log_margin(self, r: float = 0.5):
        return beta.logpdf(self.mode(), self.a, self.b) - beta.logpdf(r, self.a, self.b)


class EnsembleBeliefs(BetaDistr):
    a: Float[np.ndarray, "*copies estimators regions"]
    b: Float[np.ndarray, "*copies estimators regions"]

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

    def gather(
        self, samples_regions: Int[np.ndarray, "samples estimators"]
    ) -> Shaped[BetaDistr, "samples estimators *copies"]:
        # if multiple model copies, add trailing dimensions
        while samples_regions.ndim < self.a.ndim:
            samples_regions = samples_regions[..., None]

        # gather beliefs for each sample and estimator (and copy)
        a = np.take_along_axis(self.a.T, samples_regions, axis=0)
        b = np.take_along_axis(self.b.T, samples_regions, axis=0)
        return BetaDistr(a=a, b=b)

    def update(
        self,
        samples_regions: Int[np.ndarray, "samples estimators"],
        da: Float[np.ndarray, "samples"],
        db: Float[np.ndarray, "samples"],
    ):
        # if multiple model copies, add trailing dimensions
        while samples_regions.ndim < self.a.ndim:
            samples_regions = samples_regions[..., None]
            da = da[..., None]
            db = db[..., None]

        # gather and modify only the relevant parameters
        a_to_update, b_to_update = self.gather(samples_regions)
        a_updated = a_to_update + da[..., None]
        b_updated = b_to_update + db[..., None]

        # update the beliefs
        np.put_along_axis(self.a.T, samples_regions, a_updated, axis=0)
        np.put_along_axis(self.b.T, samples_regions, b_updated, axis=0)

    def aggregate(
        self,
        samples_regions: Int[np.ndarray, "samples estimators"],
        method: Literal["sum", "moment"],
    ) -> Shaped[BetaDistr, "samples *copies"]:
        beliefs = self.gather(samples_regions)
        if method == "sum":
            a_total = np.sum(beliefs.a, axis=1)
            b_total = np.sum(beliefs.b, axis=1)
            return BetaDistr(a=a_total, b=b_total)
        elif method == "moment":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
