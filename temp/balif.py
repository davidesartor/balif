from functools import partial
from typing import Literal, NamedTuple, Optional, Self
from jaxtyping import Array, Float, Bool, Int, PRNGKeyArray
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jax.scipy.special import gammaln
import numpy as np
from forests import IsolationTree, IsolationForest


class BetaDistribution(eqx.Module):
    alpha: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.ones(()))
    beta: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.ones(()))

    @property
    def mean(self) -> Float[Array, "..."]:
        return self.alpha / (self.alpha + self.beta)

    @property
    def nu(self) -> Float[Array, "..."]:
        return self.alpha + self.beta

    @property
    def var(self) -> Float[Array, "..."]:
        return self.alpha * self.beta / ((self.nu + 1) * (self.nu) ** 2)

    @property
    def mode(self) -> Float[Array, "..."]:
        return jax.lax.select(
            jnp.minimum(self.alpha, self.beta) > 1,
            (self.alpha - 1) / (self.nu - 2),
            jax.lax.select((self.alpha > self.beta), 1.0, 0.0),
        )

    def logpdf(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        logB = gammaln(self.alpha) + gammaln(self.beta) - gammaln(self.nu)
        log_lk = (self.alpha - 1) * jnp.log(x) + (self.beta - 1) * jnp.log(1 - x)
        return log_lk - logB

    def pdf(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return jnp.exp(self.logpdf(x))

    @classmethod
    def from_mean_and_nu(cls, mean: Float[Array, "..."], nu: Float[Array, "..."]):
        return cls(mean * nu, (1 - mean) * nu)

    @classmethod
    def from_mean_and_var(cls, mean: Float[Array, "..."], var: Float[Array, "..."]):
        alpha = mean * (mean * (1 - mean) / var - 1)
        beta = (1 - mean) * (mean * (1 - mean) / var - 1)
        return cls(alpha, beta)


class Balif(IsolationForest):
    prior_sample_size: Literal["haldane", "jeffreys", "bayes"] | float = 0.1
    score_reduction: Literal["mean", "mode"] = "mean"
    query_strategy: Literal["random", "anomalous", "margin"] = "random"

    beliefs: BetaDistribution = eqx.field(init=False, default_factory=BetaDistribution)

    @eqx.filter_jit
    def fit(self, data: Float[Array, "samples dim"], *, key: PRNGKeyArray) -> Self:
        key_fit, key_prior = jr.split(key)
        forest = super().fit(data, key=key_fit)

        def scaled_if_score(tree, node):
            correction = tree.expected_depth(tree.reached[node])
            normalization = tree.expected_depth(tree.reached[0])
            score = 2 ** (-(tree.depth(node) + correction) / normalization)
            max_depth = 1 + jnp.log2(tree.reached.shape[-1] + 1)
            min_score = 2 ** (-1 - max_depth / normalization)
            max_score = 2 ** (-1 / normalization)
            return jnp.clip((score - min_score) / (max_score - min_score), 0.001, 0.999)

        def fit_priors(tree):
            nodes = jnp.arange(tree.reached.shape[-1])
            scores = jax.vmap(scaled_if_score, in_axes=(None, 0))(tree, nodes)
            if self.prior_sample_size == "haldane":
                sample_size = 1e-8
            elif self.prior_sample_size == "jeffreys":
                sample_size = 0.5
            elif self.prior_sample_size == "bayes":
                sample_size = 1.0
            elif not isinstance(self.prior_sample_size, float):
                raise ValueError(f"Unknown prior sample size: {self.prior_sample_size}")
            else:
                sample_size = self.prior_sample_size
            sample_size = sample_size / jnp.minimum(scores, 1 - scores)
            return BetaDistribution.from_mean_and_nu(mean=scores, nu=sample_size)

        beliefs = jax.vmap(fit_priors)(forest.trees)
        return eqx.tree_at(
            lambda x: (x.trees, x.beliefs), self, (forest.trees, beliefs)
        )

    @eqx.filter_jit
    def score_as_distribution(
        self, data: Float[Array, "*batch dim"], *, key: PRNGKeyArray
    ):
        def get_belief(tree, treebeliefs, point, key) -> BetaDistribution:
            terminal, _ = tree.path(point, key=key)
            belief = jax.tree.map(lambda x: x[terminal], treebeliefs)
            return belief

        def combined_beliefs(point):
            keys = jr.split(key, self.n_estimators)
            beliefs = jax.vmap(get_belief, in_axes=(0, 0, None, 0))(
                self.trees, self.beliefs, point, keys
            )
            return BetaDistribution.from_mean_and_var(
                mean=beliefs.mean.mean(),
                var=beliefs.var.mean() / self.n_estimators,
            )

        *batch, dim = data.shape
        for _ in batch:
            combined_beliefs = jax.vmap(combined_beliefs)
        return combined_beliefs(data)

    @eqx.filter_jit
    def score(self, data: Float[Array, "*batch dim"], *, key: PRNGKeyArray):
        belief = self.score_as_distribution(data, key=key)
        if self.score_reduction == "mean":
            return belief.mean
        elif self.score_reduction == "mode":
            return belief.mode
        else:
            raise ValueError(f"Unknown score reduction method: {self.score_reduction}")

    @eqx.filter_jit
    def register(
        self,
        data: Float[Array, "*batch dim"],
        *,
        key: PRNGKeyArray,
        is_anomaly: Bool[Array, "*batch"],
    ) -> Self:
        def update_tree(tree, treebeliefs, point, is_anom, key):
            terminal, path = tree.path(point, key=key)
            new_beliefs = jax.tree.map(
                lambda b, obs: b.at[terminal].add(obs),
                treebeliefs,
                BetaDistribution(is_anom, 1 - is_anom),
            )
            return new_beliefs

        def update_forest(point, is_anom):
            keys = jr.split(key, self.n_estimators)
            return jax.vmap(update_tree, in_axes=(0, 0, None, None, 0))(
                self.trees, self.beliefs, point, is_anom, keys
            )

        *batch, dim = data.shape
        for _ in batch:
            update_forest = jax.vmap(update_forest)
        new_beliefs = update_forest(data, is_anomaly)
        return eqx.tree_at(lambda x: x.beliefs, self, new_beliefs)

    @eqx.filter_jit
    def interest(
        self,
        data: Float[Array, "*batch dim"],
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, "*batch"]:
        def margin(point):
            r = jnp.array(0.5)
            belief = self.score_as_distribution(point, key=key)
            logmargin = belief.logpdf(belief.mode) - belief.logpdf(r)
            return jnp.exp(-logmargin)

        if self.query_strategy == "margin":
            interest_fn = margin
        elif self.query_strategy == "anomalous":
            interest_fn = lambda p: self.score(p, key=key)
        elif self.query_strategy == "random":
            interest_fn = lambda p: jr.uniform(key)
        else:
            raise ValueError(f"Unknown query strategy: {self.query_strategy}")

        *batch, dim = data.shape
        for _ in batch:
            interest_fn = jax.vmap(interest_fn)
        return interest_fn(data)
