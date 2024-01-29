from functools import partial
from typing import Optional
import jax
import jax.numpy as jnp
from flax import struct
from isolation_tree import IsolationTree


def expected_depth(sizes: int | jax.Array) -> jax.Array:
    """Compute the expected isolation depth for a node with n data points."""

    def correction(n: int | jax.Array) -> jax.Array:
        EULER_MASCHERONI = 0.5772156649
        harm_number = jnp.log(n - 1) + EULER_MASCHERONI
        expected_depth = 2 * harm_number - 2 * (n - 1) / n
        return expected_depth

    return jnp.where(sizes <= 1, 0.0, correction(sizes))


class ExtendedIsolationForest(struct.PyTreeNode):
    normals: jax.Array
    intercepts: jax.Array
    node_sizes: jax.Array
    expected_subtree_depth: jax.Array

    @jax.jit
    def paths(self, point: jax.Array) -> jax.Array:
        paths = jax.vmap(IsolationTree.path, in_axes=(0, None))(self, point)
        return paths

    @jax.jit
    def score(self, point: jax.Array) -> jax.Array:
        paths = self.paths(point)
        sizes = jax.vmap(jnp.take)(self.node_sizes, paths)
        corrections = jax.vmap(jnp.take)(self.expected_subtree_depth, paths)

        isolation_depths = sizes.argmin(axis=-1) + corrections.min(axis=-1)
        score = 2 ** -(isolation_depths / corrections.max(axis=-1)).mean()
        return score

    @jax.jit
    def score_samples(self, data: jax.Array) -> jax.Array:
        return jax.vmap(self.score)(data)

    def decision_function(self, data: jax.Array) -> jax.Array:
        ...

    def predict(self, data: jax.Array) -> jax.Array:
        ...

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
    ):
        rng_subsample, rng_forest = jax.random.split(rng)
        max_samples = max_samples or min((256, data.shape[0]))

        subsamples = jax.vmap(
            partial(jax.random.choice, a=data, shape=(max_samples,), replace=bootstrap)
        )(jax.random.split(rng_subsample, n_estimators))

        trees = jax.vmap(IsolationTree.fit, in_axes=(0, 0, None))(
            jax.random.split(rng_forest, n_estimators),
            subsamples,
            hyperplane_components or data.shape[-1],
        )
        return cls(
            trees.normals, trees.intercepts, trees.node_sizes, expected_depth(trees.node_sizes)
        )
