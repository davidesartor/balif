from typing import Literal, Optional, NamedTuple
from jaxtyping import Float, Int, Shaped

import numpy as np
from scipy.stats import beta

from pyod.models.base import BaseDetector


class BetaDistr(NamedTuple):
    a: Float[np.ndarray, "..."]
    b: Float[np.ndarray, "..."]

    def mu(self):
        return self.a / (self.a + self.b)

    def ss(self):
        return self.a + self.b

    def mode(self):
        return np.where(
            np.minimum(self.a, self.b) > 1,
            (self.a - 1) / (self.a + self.b - 2),
            (self.a > self.b).astype(float),
        )

    def log_pdf(self, x):
        return beta.logpdf(x, self.a, self.b)


class BayesianDetector(BaseDetector):
    @property
    def regions_score(self) -> Float[np.ndarray, "estimators regions"]:
        raise NotImplementedError

    def estimators_apply(
        self, X: Float[np.ndarray, "samples features"]
    ) -> Int[np.ndarray, "estimators samples"]:
        raise NotImplementedError

    def __init__(
        self,
        *args,
        prior_sample_size: float = 0.1,
        mu_aggregation_method: Literal["avg", "geom"] = "avg",
        ss_aggregation_method: Literal["sum", "moment"] = "sum",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prior_sample_size = prior_sample_size
        self.mu_aggregation_method = mu_aggregation_method
        self.ss_aggregation_method = ss_aggregation_method
        self.beliefs: Shaped[BetaDistr, "estimators regions"]  # call .fit to initialize

    def recompute_threshold(self, X: Float[np.ndarray, "samples features"]):
        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()

    def fit(
        self,
        X: Float[np.ndarray, "samples features"],
        y: Optional[Int[np.ndarray, "samples 1"]] = None,
    ):
        # fit unsupervised model
        super().fit(X, y)

        # flat prior, matching the expected contamination
        prior_a = self.contamination * self.prior_sample_size
        prior_b = (1 - self.contamination) * self.prior_sample_size

        # add positive obs matching the mean to detector scores
        regions_score = np.clip(self.regions_score, 0.01, 0.99)
        a_over_b = regions_score / (1 - regions_score)
        self.beliefs = BetaDistr(
            a=np.maximum(a_over_b * prior_b, prior_a),
            b=np.maximum(prior_a / a_over_b, prior_b),
        )

        self.recompute_threshold(X)
        return self

    def gather_beliefs(
        self, regions: Int[np.ndarray, "etimators samples"]
    ) -> Shaped[BetaDistr, "estimators samples"]:
        return BetaDistr(
            a=np.take_along_axis(self.beliefs.a, regions, axis=-1),
            b=np.take_along_axis(self.beliefs.b, regions, axis=-1),
        )

    def aggregate_beliefs(
        self, beliefs: Shaped[BetaDistr, "estimators samples"]
    ) -> Shaped[BetaDistr, "samples"]:
        if self.mu_aggregation_method == "avg":
            mu = np.mean(beliefs.mu(), axis=0)
        elif self.mu_aggregation_method == "geom":
            mu = np.exp(np.mean(np.log(beliefs.mu()), axis=0))
        else:
            raise ValueError(f"Unknown mean aggregation method: {self.mu_aggregation_method}")

        if self.ss_aggregation_method == "sum":
            ss = np.sum(beliefs.ss(), axis=0)
        elif self.ss_aggregation_method == "moment":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown aggregation method: {self.ss_aggregation_method}")

        return BetaDistr(a=mu * ss, b=(1 - mu) * ss)

    def decision_function(
        self, X: Float[np.ndarray, "samples features"]
    ) -> Float[np.ndarray, "samples"]:
        regions = self.estimators_apply(X)
        beliefs = self.gather_beliefs(regions)
        beliefs = self.aggregate_beliefs(beliefs)
        return beliefs.mu()

    def update(
        self,
        X: Float[np.ndarray, "samples features"],
        y: Int[np.ndarray, "samples"],
        confidence: float | Float[np.ndarray, "#samples"] = 1.0,
    ):
        da = confidence * (y >= 1)
        db = confidence * (y == 0)
        regions = self.estimators_apply(X)

        # updata one point at a time to avoid overwriting
        # (TODO: do this in 1 step aggregating points in the same region)
        for i in range(regions.shape[-1]):
            regions_sample_i = regions[..., i][..., None]
            a, b = self.gather_beliefs(regions_sample_i)
            np.put_along_axis(self.beliefs.a, regions_sample_i, a + da[i], axis=-1)
            np.put_along_axis(self.beliefs.b, regions_sample_i, b + db[i], axis=-1)
