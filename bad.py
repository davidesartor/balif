from typing import Literal, Optional, NamedTuple
from jaxtyping import Float, Int, Bool, Shaped

import numpy as np
from scipy.stats import beta
from scipy.special import digamma
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
        reprocess_decision_scores: bool = True,
        prior_sample_size: float = 0.1,
        aggregation_method: Literal["sum", "moment"] = "sum",
        interest_method: Literal["margin", "anom", "bald"] = "margin",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prior_sample_size = prior_sample_size
        self.aggregation_method: Literal["sum", "moment"] = aggregation_method
        self.interest_method: Literal["bald", "margin", "anom"] = interest_method
        self.reprocess_decision_scores = reprocess_decision_scores

        self.beliefs: Shaped[BetaDistr, "estimators regions"]  # call .fit to initialize

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

        if self.reprocess_decision_scores:
            self.decision_scores_ = self.decision_function(X)
            self._process_decision_scores()
        return self

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

    def get_queries(
        self,
        X: Float[np.ndarray, "samples features"],
        batch_size: int = 1,
        independent: bool = True,
        mask: Optional[Bool[np.ndarray, "samples"]] = None,
    ) -> Int[np.ndarray, "batch"]:
        # initialize the mask if not provided
        if mask is None:
            mask = np.ones(X.shape[0], dtype=bool)
        mask = mask.copy()

        # gather the beliefs for each estimator and sample
        regions = self.estimators_apply(X)
        beliefs = self.gather_beliefs(regions)

        # if independent or batch_size == 1 directly return top k samples
        if independent or batch_size == 1:
            scores = np.where(mask, self.interest(beliefs), -np.inf)
            queries_idxs = scores.argsort()[-batch_size:]
            return queries_idxs

        queries_idxs = []
        for i in range(batch_size):
            # query most interesting point
            scores = np.where(mask, self.interest(beliefs), -np.inf)
            queries_idxs.append(scores.argmax())
            mask[queries_idxs[-1]] = False

            # worst case beliefs for the queried point
            if self.interest_method in ["margin", "bald"]:
                mu = self.aggregate_beliefs(beliefs).mu()
                worstcase_beliefs = BetaDistr(
                    a=beliefs.a + (mu > 0.5).astype(float),
                    b=beliefs.b + (mu <= 0.5).astype(float),
                )
            elif self.interest_method == "anom":
                worstcase_beliefs = BetaDistr(a=beliefs.a, b=beliefs.b + 1.0)
            else:
                raise ValueError(f"Unknown interest method: {self.interest_method}")

            # update the beliefs of the affected regions
            affected_regions = regions == regions[:, queries_idxs[-1]][..., None]
            beliefs = BetaDistr(
                a=np.where(affected_regions, worstcase_beliefs.a, beliefs.a),
                b=np.where(affected_regions, worstcase_beliefs.b, beliefs.b),
            )
        return np.array(queries_idxs)

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
        if self.aggregation_method == "sum":
            mu = np.mean(beliefs.mu(), axis=0)
            ss = np.sum(beliefs.ss(), axis=0)
            return BetaDistr(a=mu * ss, b=(1 - mu) * ss)
        elif self.aggregation_method == "moment":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def interest(
        self, beliefs: Shaped[BetaDistr, "estimators samples"]
    ) -> Float[np.ndarray, "samples"]:
        beliefs = self.aggregate_beliefs(beliefs)
        if self.interest_method == "margin":
            log_margin = beliefs.log_pdf(beliefs.mode()) - beliefs.log_pdf(0.5)
            return np.exp(-log_margin)
        elif self.interest_method == "anom":
            return beliefs.mu()
        elif self.interest_method == "bald":
            a, b, mu = beliefs.a, beliefs.b, beliefs.mu()
            H_y = -mu * np.log(mu) - (1 - mu) * np.log(1 - mu)
            H_y_given_w = digamma(a + b + 1) - mu * digamma(a + 1) - (1 - mu) * digamma(b + 1)
            I_yw = H_y - H_y_given_w
            return I_yw
        else:
            raise ValueError(f"Unknown interest method: {self.interest_method}")
