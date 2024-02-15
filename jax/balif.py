from functools import partial
from typing import Literal, Optional, overload
import jax
import jax.numpy as jnp
from flax import struct
from isolation_forest import IsolationTree, ExtendedIsolationForest


class BalifTree(IsolationTree):
    alphas: jax.Array  # shape (nodes,)
    betas: jax.Array  # shape (nodes,)

    @jax.jit
    def score(self, point: jax.Array) -> jax.Array:
        path = self.path(point)
        isolation_node = path[self.node_sizes[path].argmin()]
        alpha = self.alphas[isolation_node]
        beta = self.betas[isolation_node]
        return alpha / (alpha + beta)

    @jax.jit
    def score_full_path(self, point: jax.Array) -> jax.Array:
        path = self.path(point)
        alpha = jnp.mean(self.alphas[path] * 2 ** jnp.arange(path.size))
        beta = jnp.mean(self.betas[path] * 2 ** jnp.arange(path.size))
        return alpha / (alpha + beta)

    @jax.jit
    def register(self, point: jax.Array, is_anomaly: bool):
        path = self.path(point)
        new_alphas = jax.lax.select(is_anomaly, self.alphas.at[path].add(1), self.alphas)
        new_betas = jax.lax.select(is_anomaly, self.betas, self.betas.at[path].add(1))
        return self.replace(alphas=new_alphas, betas=new_betas)


class Balif(struct.PyTreeNode):
    trees: BalifTree

    @jax.jit
    def score(self, point: jax.Array, use_full_path=True) -> jax.Array:
        tree_scores = jax.vmap(BalifTree.score, in_axes=(0, None, None))(
            self.trees, point, use_full_path
        )
        return jnp.exp(jnp.log(tree_scores).mean())

    @jax.jit
    def score_samples(self, data: jax.Array, use_full_path=True) -> jax.Array:
        return jax.vmap(self.score, in_axes=(0, None))(data, use_full_path)

    @jax.jit
    def register(self, point: jax.Array, is_anomaly: bool):
        vmap_register = jax.vmap(BalifTree.register, in_axes=(0, None, None))
        return self.replace(trees=vmap_register(self.trees, point, is_anomaly))

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
        forest: ExtendedIsolationForest = ExtendedIsolationForest.fit(
            rng, data, n_estimators, max_samples, hyperplane_components, bootstrap
        )

        # single tree IF score. Note that this is not a good way to set the prior
        # the root has score of 0.5 -> contamination = 0.5
        # a node separated at first step has score of 2**(1/(2log(psi)-1)) ~= 0.95
        # a node at maxdepth with all points has score of 2**-1 -log2(psi)/(2log(psi)-1) ~= 0.3
        corrected_depth = forest.trees.depths + forest.trees.expected_subtree_depth
        expected_from_root = forest.trees.expected_subtree_depth[:, 0]
        base_scores = 2 ** -(corrected_depth / expected_from_root[:, None])

        # naive range correction
        base_scores = (4 / 3) * (base_scores - 0.25)

        # clips prediction for "phantom" nodes after isolation
        # this is just an approximation, and sets it to the most extreme value in the tree
        isolation_scores = jnp.max(base_scores, axis=-1, keepdims=True)
        base_scores = jnp.where(forest.trees.node_sizes > 1, base_scores, isolation_scores)

        # match the predition adding strictly positive virtual samples
        prior_sample_size = 0.1
        sample_size_after_IF = prior_sample_size / jnp.minimum(base_scores, 1 - base_scores)
        alphas = sample_size_after_IF * base_scores
        betas = sample_size_after_IF * (1 - base_scores)

        return cls(
            BalifTree(
                forest.trees.normals,
                forest.trees.intercepts,
                forest.trees.depths,
                forest.trees.node_sizes,
                forest.trees.expected_subtree_depth,
                alphas,
                betas,
            )
        )
