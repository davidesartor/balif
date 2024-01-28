from functools import partial
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


ForestState = IsolationTree  # isolation tree where fields have a batch dimension


class ExtendedIsolationForest(struct.PyTreeNode):
    n_estimators: int
    max_samples: int
    hyperplane_components: int | None = None
    bootstrap: bool = True

    @partial(jax.jit, static_argnames=("self",))
    def fit(self, rng: jax.Array, data: jax.Array) -> ForestState:
        rng_subsample, rng_forest = jax.random.split(rng)

        subsamples = jax.vmap(
            partial(jax.random.choice, a=data, shape=(self.max_samples,), replace=self.bootstrap)
        )(jax.random.split(rng_subsample, self.n_estimators))

        return jax.vmap(IsolationTree.fit, in_axes=(0, 0, None))(
            jax.random.split(rng_forest, self.n_estimators),
            subsamples,
            self.hyperplane_components or data.shape[-1],
        )

    @jax.jit
    def score_samples(self, trees: ForestState, data: jax.Array) -> jax.Array:
        get_paths_size = jax.vmap(IsolationTree.path_sizes, in_axes=(None, 0))
        get_forest_paths_size = jax.vmap(get_paths_size, in_axes=(0, None))

        path_sizes = get_forest_paths_size(trees, data)
        isolation_depths = path_sizes.argmin(axis=-1) + expected_depth(path_sizes.min(axis=-1))
        score = jnp.power(2, -isolation_depths.mean(axis=0) / expected_depth(self.max_samples))
        return score

    def decision_function(self, data: jax.Array) -> jax.Array:
        ...

    def predict(self, data: jax.Array) -> jax.Array:
        ...
