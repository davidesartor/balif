from functools import partial
from typing import Literal, Optional, overload
import jax
import jax.numpy as jnp
from flax import struct
from isolation_tree import IsolationTree
from isolation_forest import ExtendedIsolationForest


class Balif(ExtendedIsolationForest):
    alphas: jax.Array
    betas: jax.Array

    @jax.jit
    def score(self, point: jax.Array):
        alpha, beta = self.get_distr_params(point)
        score = alpha / (alpha + beta)
        score = jnp.exp(jnp.mean(jnp.log(score)))  # geometric mean
        return score

    @jax.jit
    def get_distr_params(self, point: jax.Array) -> tuple[jax.Array, jax.Array]:
        paths = self.paths(point)
        isolation_depths = jax.vmap(jnp.take)(self.node_sizes, paths).argmin(axis=-1)
        isolation_nodes = jax.vmap(jnp.take)(paths, isolation_depths)
        alpha = jax.vmap(jnp.take)(self.alphas, isolation_nodes)
        beta = jax.vmap(jnp.take)(self.betas, isolation_nodes)
        return alpha, beta

    @partial(jax.jit, static_argnums=(-1,))
    def interest_for(self, point: jax.Array, strat="margin") -> jax.Array:
        if strat == "anom":
            return self.score(point)
        elif strat == "margin":
            alpha, beta = self.get_distr_params(point)
            # alpha beta sono le distr di ogni nodo, il margine è globale
            # da sistemare una volta decisa la regola di riduzione
            r = 0.5
            margin = jax.scipy.stats.beta.pdf(r, alpha, beta)
            return margin
        else:
            raise ValueError(f"Unknown interest strategy {strat}")

    @jax.jit
    def update(self, point: jax.Array, is_anomaly=True):
        def tree_update(alphas, betas, path):
            new_alphas = jax.lax.select(is_anomaly, alphas.at[path].add(1), alphas)
            new_betas = jax.lax.select(is_anomaly, betas, betas.at[path].add(1))
            return new_alphas, new_betas

        new_alphas, new_betas = jax.vmap(tree_update)(self.alphas, self.betas, self.paths(point))
        return self.replace(alphas=new_alphas, betas=new_betas)

    @classmethod
    @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6))
    def fit(
        cls,
        rng: jax.Array,
        data: jax.Array,
        n_estimators: int,
        max_samples: Optional[int] = None,
        hyperplane_components: Optional[int] = None,
        bootstrap: bool = True,
        prior_sample_size=1.0,
    ):
        forest = ExtendedIsolationForest.fit(
            rng, data, n_estimators, max_samples, hyperplane_components, bootstrap
        )
        # single tree IF score. Note that this is not a good way to set the prior
        # the root has score of 0.5 -> contamination = 0.5
        # a node separated at first step has score of 2**(1/(2log(psi)-1)) ~= 0.95
        # a node at maxdepth with all points has score of 2**-1 -log2(psi)/(2log(psi)-1) ~= 0.3
        node_depth = jnp.log2(1 + jnp.indices(forest.node_sizes.shape)[1]).astype(int)
        expected_from_root = forest.expected_subtree_depth.max(axis=-1, keepdims=True)
        scores = 2 ** -((node_depth + forest.expected_subtree_depth) / expected_from_root)

        # naive correction
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        alphas = prior_sample_size * scores
        betas = prior_sample_size - alphas

        return cls(
            forest.normals,
            forest.intercepts,
            forest.node_sizes,
            forest.expected_subtree_depth,
            alphas,
            betas,
        )
