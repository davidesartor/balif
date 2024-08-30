from functools import partial
from typing import Literal, Optional, NamedTuple
from jaxtyping import Float, Int, Shaped

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import beta
import equinox as eqx

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
            regions_score=jnp.asarray(self.regions_score),
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
        regions = jnp.asarray(self.estimators_apply(X))
        scores = self.beliefs.aggregate(regions, self.aggregation_method)
        return np.asarray(scores)

    def update(
        self,
        X: Float[np.ndarray, "samples features"],
        y: Int[np.ndarray, "samples 1"],
        confidence: float | Float[np.ndarray, "#samples"] = 1.0,
    ):
        regions = jnp.asarray(self.estimators_apply(X))
        da = jnp.asarray(confidence * (y >= 1)).flatten()
        db = jnp.asarray(confidence * (y == 0)).flatten()
        self.beliefs = self.beliefs.update(regions, da, db)


class BetaDistr(eqx.Module):
    a: Float[jax.Array, "..."]
    b: Float[jax.Array, "..."]

    def mean(self):
        return self.a / (self.a + self.b)


class EnsembleBeliefs(BetaDistr):
    a: Float[jax.Array, "estimators regions"]
    b: Float[jax.Array, "estimators regions"]

    @classmethod
    @eqx.filter_jit
    def from_scores(
        cls,
        regions_score: Float[jax.Array, "estimators regions"],
        contamination: float = 0.1,
        prior_sample_size: float = 0.1,
    ):
        # flat prior, matching the contamination
        prior_a = contamination * prior_sample_size
        prior_b = (1 - contamination) * prior_sample_size

        # add positive obs matching the mean to detector scores
        regions_score = jnp.clip(regions_score, 0.01, 0.99)
        a_over_b = regions_score / (1 - regions_score)
        a = jnp.maximum(a_over_b * prior_b, prior_a)
        b = jnp.maximum(prior_a / a_over_b, prior_b)
        return cls(a=a, b=b)

    @eqx.filter_jit
    def update(
        self,
        samples_regions: Int[jax.Array, "samples estimators"],
        da: Float[jax.Array, "samples"],
        db: Float[jax.Array, "samples"],
    ):
        def single_update(beliefs, region, da, db):
            new_a = beliefs.a.at[region].add(da)
            new_b = beliefs.b.at[region].add(db)
            return eqx.tree_at(lambda t: (t.a, t.b), beliefs, (new_a, new_b))

        def scan_fn(beliefs, update_info):
            update_all = jax.vmap(single_update, in_axes=(0, 0, None, None))
            return update_all(beliefs, *update_info), None

        self, _ = jax.lax.scan(scan_fn, self, (samples_regions, da, db))
        return self

    @eqx.filter_jit
    def gather(
        self, samples_regions: Int[jax.Array, "samples estimators"]
    ) -> Shaped[BetaDistr, "samples estimators"]:
        def take(distr, idx):
            return BetaDistr(a=distr.a[idx], b=distr.b[idx])

        take = jax.vmap(take, in_axes=(0, 0))  # map over estimators
        take = jax.vmap(take, in_axes=(None, 0))  # map over samples
        return take(self, samples_regions)

    @eqx.filter_jit
    def aggregate(
        self, samples_regions: Int[jax.Array, "samples estimators"], method: str
    ) -> Float[jax.Array, "samples"]:
        beliefs = self.gather(samples_regions)

        if method == "arithmetic_mean":
            return jnp.mean(beliefs.mean(), axis=-1)
        elif method == "geometric_mean":
            return np.exp(np.mean(np.log(beliefs.mean()), axis=-1))
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
