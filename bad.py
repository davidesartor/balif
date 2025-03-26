from typing import Literal, Optional, NamedTuple
from jaxtyping import Float, Int, Shaped


import copy
import numpy as np
from scipy.stats import beta
from pyod.models.base import BaseDetector


class BetaDistr(NamedTuple):
    a: Float[np.ndarray, "..."]
    b: Float[np.ndarray, "..."]

    def mean(self):
        return self.a / (self.a + self.b)

    def samplesize(self):
        return self.a + self.b

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
        self, regions: Int[np.ndarray, "estimators samples"]
    ) -> Shaped[BetaDistr, "*copies estimators samples"]:
        # if multiple model copies, add leading dimensions
        while regions.ndim < self.a.ndim:
            regions = regions[None, ...]

        # gather beliefs for each sample and estimator (and copy)
        a = np.take_along_axis(self.a, regions, axis=-1)
        b = np.take_along_axis(self.b, regions, axis=-1)
        return BetaDistr(a=a, b=b)

    def update(
        self,
        regions: Int[np.ndarray, "estimators samples"],
        da: Float[np.ndarray, "samples"],
        db: Float[np.ndarray, "samples"],
    ):
        # if multiple model copies, add leading dimensions
        while regions.ndim < self.a.ndim:
            regions = regions[None, ...]

        # updata one point at a time to avoid overwriting
        for i in range(regions.shape[-1]):
            sample_regions = regions[..., i : i + 1]
            a, b = self.gather(sample_regions)
            np.put_along_axis(self.a, sample_regions, a + da[i], axis=-1)
            np.put_along_axis(self.b, sample_regions, b + db[i], axis=-1)

    def aggregate(
        self,
        regions: Int[np.ndarray, "estimators samples"],
        method: Literal["sum", "moment"],
    ) -> Shaped[BetaDistr, "*copies samples"]:
        beliefs = self.gather(regions)
        if method == "sum":
            mu = np.mean(beliefs.mean(), axis=-2)
            ss = np.sum(beliefs.samplesize(), axis=-2)
            a_total = mu * ss
            b_total = (1 - mu) * ss
            return BetaDistr(a=a_total, b=b_total)
        elif method == "moment":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


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
        interest_method: Literal["margin", "anom"] = "margin",
        batch_query_method: Literal[
            "worstcase", "average", "bestcase", "independent"
        ] = "independent",
        
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prior_sample_size = prior_sample_size
        self.aggregation_method: Literal["sum", "moment"] = aggregation_method
        self.interest_method: Literal["margin", "anom"] = interest_method
        self.reprocess_decision_scores = reprocess_decision_scores
        self.batch_query_method: Literal[
            "worstcase", "average", "bestcase", "independent"
        ] = batch_query_method

    def fit(
        self,
        X: Float[np.ndarray, "samples features"],
        y: Optional[Int[np.ndarray, "samples 1"]] = None,
    ):
        super().fit(X, y)
        self.ensemble_beliefs = EnsembleBeliefs.from_scores(
            regions_score=self.regions_score,
            contamination=self.contamination,
            prior_sample_size=self.prior_sample_size,
        )
        if self.reprocess_decision_scores:
            self.decision_scores_ = self.decision_function(X)
            self._process_decision_scores()
        return self

    def decision_function(
        self,
        X: Float[np.ndarray, "samples features"],
    ) -> Float[np.ndarray, "samples"]:
        regions = self.estimators_apply(X)
        beliefs = self.ensemble_beliefs.aggregate(regions, self.aggregation_method)
        scores = beliefs.mean()
        return scores

    def update(
        self,
        X: Float[np.ndarray, "samples features"],
        y: Int[np.ndarray, "samples"],
        confidence: float | Float[np.ndarray, "#samples"] = 1.0,
    ):
        da = confidence * (y >= 1)
        db = confidence * (y == 0)
        regions = self.estimators_apply(X)
        self.ensemble_beliefs.update(regions, da, db)

    def interest(
        self,
        X: Float[np.ndarray, "samples features"],
        ensemble_beliefs: Optional[EnsembleBeliefs] = None,
    ) -> Float[np.ndarray, "*copies samples"]:
        regions = self.estimators_apply(X)
        ensemble_beliefs = ensemble_beliefs or self.ensemble_beliefs
        beliefs = ensemble_beliefs.aggregate(regions, method=self.aggregation_method)
        if self.interest_method == "margin":
            return np.exp(-beliefs.log_margin())
        elif self.interest_method == "anom":
            return beliefs.mean()
        else:
            raise ValueError(f"Unknown interest method: {self.interest_method}")

    def get_queries(
        self,
        X: Float[np.ndarray, "samples features"],
        batch_size: int = 1,
    ) -> Int[np.ndarray, "batch_size"]:
        if self.batch_query_method == "independent":
            idxs = self.interest(X).argsort()[-batch_size:]
            return idxs

        # initialize the superposition model
        beliefs_superposition = EnsembleBeliefs(
            a=self.ensemble_beliefs.a[None, ...].copy(),
            b=self.ensemble_beliefs.b[None, ...].copy(),
        )

        queries_idx = []
        queriable = np.ones(X.shape[0], dtype=bool)
        weights = np.ones((1, 1))
        c = self.contamination

        for i in range(batch_size):
            # compute the interest of each sample
            if self.batch_query_method == "worstcase":
                interest = self.interest(X, beliefs_superposition)
                interest = interest.min(axis=0)
            elif self.batch_query_method == "bestcase":
                interest = self.interest(X, beliefs_superposition)
                interest = interest.max(axis=0)
            elif self.batch_query_method == "average":
                interest = self.interest(X, beliefs_superposition)
                interest = np.sum(interest * weights, axis=0)
                weights = np.concatenate([weights * c, weights * (1 - c)])
            else:
                raise ValueError(f"Unknown method: {self.batch_query_method}")

            # query the most interesting sample
            query_idx = np.where(queriable, interest, -np.inf).argmax()
            queriable[query_idx] = False
            queries_idx.append(query_idx)

            # update the superposition model
            regions = self.estimators_apply(X[query_idx].reshape(1, -1))
            other_superposition = copy.deepcopy(beliefs_superposition)
            beliefs_superposition.update(regions, da=np.ones(1), db=np.zeros(1))
            other_superposition.update(regions, da=np.zeros(1), db=np.ones(1))
            beliefs_superposition = EnsembleBeliefs(
                a=np.concatenate([beliefs_superposition.a, other_superposition.a]),
                b=np.concatenate([beliefs_superposition.b, other_superposition.b]),
            )
        return np.array(queries_idx)
