from typing_extensions import Self
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np


def sample_vector(rng: jax.Array, dim: int, non_zero_components: int):
    rng_value, rng_idx = jax.random.split(rng)
    if non_zero_components == 1:
        idx = jax.random.randint(rng_idx, (), 0, dim)
        return jnp.zeros((dim,)).at[idx].set(1.0)

    idxs = jax.random.choice(rng_idx, dim, (non_zero_components,), replace=False)
    values = jax.random.normal(rng_value, shape=(non_zero_components,))
    vector = jnp.zeros((dim,)).at[idxs].set(values)
    return vector / jnp.linalg.norm(vector)


def sample_intercept(rng: jax.Array, distances: jax.Array, mask: jax.Array):
    min_distance = jnp.where(mask, distances, jnp.inf).min()
    max_distance = jnp.where(mask, distances, -jnp.inf).max()
    intercept = jax.random.uniform(rng, (), minval=min_distance, maxval=max_distance)
    return intercept


class IsolationTree(NamedTuple):
    node_sizes: jax.Array
    normals: jax.Array
    intercepts: jax.Array

    @property
    def max_depth(self) -> int:
        return int(np.log2(self.intercepts.shape[-1] + 1) - 1)

    def path(self, point: jax.Array) -> jax.Array:
        def scan_body(id, x):
            distance = jnp.dot(self.normals[id], point) - self.intercepts[id]
            child = jax.lax.select(distance >= 0, 2 * id + 1, 2 * id + 2)
            return child, id

        root = jnp.zeros((), dtype=int)
        leaf, path = jax.lax.scan(scan_body, root, None, length=self.max_depth - 1)
        return jnp.concatenate((path, leaf[None,]))

    def path_sizes(self, point: jax.Array) -> jax.Array:
        return self.node_sizes[self.path(point)]

    @classmethod
    def fit(cls, rng: jax.Array, data: jax.Array, hyperplane_components: int) -> Self:
        points, features = data.shape
        max_depth = np.ceil(np.log2(points)).astype(np.int8)
        nodes = 2 ** (max_depth + 1) - 1
        rng_normal, rng_intercept = jax.random.split(rng)

        # for each node, initialize intercepts to 0.0 and presample normal vector
        intercepts = jnp.zeros((nodes,))
        sample_normal_batch = jax.vmap(sample_vector, in_axes=(0, None, None))
        normals = sample_normal_batch(
            jax.random.split(rng_normal, nodes), features, hyperplane_components
        )
        # precompute dot products between data points and normals
        # doing it this way technically requires more compute but
        # it is faster in practice, especially on GPU due metter memory patterns
        dot_prod_batch = jax.vmap(jax.vmap(jnp.dot, in_axes=(0, None)), in_axes=(None, 0))
        distances = dot_prod_batch(data, normals)

        # keep track of which nodes have been reached by each data point
        # splitting with a single point "forks" the path, so this needs to be
        # bool of shape (nodes, points) instead of int of shape (depth, points).
        # at the start, only the root node has been reached by all points
        reached = jnp.zeros((nodes, points), dtype=jnp.bool_)
        reached = reached.at[0, :].set(True)

        for depth, rng_layer in enumerate(jax.random.split(rng_intercept, max_depth)):
            # now we can iterate over the tree nodes, starting from the root
            # updating the intercept with one that splits the data points
            # and then update accordingly the reached mask for the children nodes.
            # All this can be done in parallel for all nodes at a given depth, so
            # we loop over depth instead of nodes, and perform vectorized operations.
            # the nodes idx at a given depth d are: {2**d - 1, 2**d, ..., 2**d - 1 + 2**d}
            layer_slice = jnp.arange(2**depth - 1, 2**depth - 1 + 2**depth)

            # update the intercepts to make the plane split the data points
            intercepts = intercepts.at[layer_slice].set(
                jax.vmap(sample_intercept)(
                    jax.random.split(rng_layer, 2**depth),
                    distances[layer_slice],
                    reached[layer_slice],
                )
            )

            # update the reached masks for the children nodes
            reached = reached.at[2 * layer_slice + 1, :].set(
                reached[layer_slice] & (distances[layer_slice] >= intercepts[layer_slice, None])
            )
            reached = reached.at[2 * layer_slice + 2, :].set(
                reached[layer_slice] & (distances[layer_slice] <= intercepts[layer_slice, None])
            )
        return cls(reached.sum(axis=1), normals, intercepts)
