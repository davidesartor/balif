from functools import partial
from typing import Optional
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def expected_depth(node_sizes: jax.Array) -> jax.Array:
    """Compute the expected isolation depth for a node with n data points."""

    def correction(n: jax.Array) -> jax.Array:
        EULER_MASCHERONI = 0.5772156649
        harm_number = jnp.log(n - 1) + EULER_MASCHERONI
        expected_depth = 2 * harm_number - 2 * (n - 1) / n
        return expected_depth

    return jnp.where(node_sizes <= 1, 0.0, correction(node_sizes))


@partial(jax.jit, static_argnames=("dim", "non_zero_components"))
def sample_vector(rng: jax.Array, dim: int, non_zero_components: int) -> jax.Array:
    """Sample a (possibly sparse) vector uniformly on the unit hypersphere."""
    rng_value, rng_idx = jax.random.split(rng)
    idxs = jax.random.choice(rng_idx, dim, (non_zero_components,), replace=False)
    values = jax.random.normal(rng_value, shape=(non_zero_components,))
    vector = jnp.zeros((dim,)).at[idxs].set(values)
    return vector / jnp.linalg.norm(vector)


@jax.jit
def sample_intercept(rng: jax.Array, distances: jax.Array, mask: jax.Array):
    """Sample an intercept that splits the data points."""
    min_distance = jnp.where(mask, distances, jnp.inf).min()
    max_distance = jnp.where(mask, distances, -jnp.inf).max()
    intercept = jax.random.uniform(rng, (), minval=min_distance, maxval=max_distance)
    return intercept


class IsolationTree(struct.PyTreeNode):
    normals: jax.Array  # shape (nodes, features)
    intercepts: jax.Array  # shape (nodes,)
    depths: jax.Array  # shape (nodes,)
    node_sizes: jax.Array  # shape (nodes,)
    expected_subtree_depth: jax.Array  # shape (nodes,)

    @jax.jit
    def path(self, point: jax.Array) -> jax.Array:
        """Return the path for a given point in the tree as an Array of node idx."""

        def scan_body(id, x):
            distance = jnp.dot(self.normals[id], point) - self.intercepts[id]
            child = jax.lax.select(distance >= 0, 2 * id + 1, 2 * id + 2)
            return child, id

        root = jnp.zeros((), dtype=int)
        max_depth = int(np.log2(self.node_sizes.size + 1)) - 1
        _, path = jax.lax.scan(scan_body, root, None, length=max_depth)
        return path

    @jax.jit
    def score(self, point: jax.Array) -> jax.Array:
        """Return the isolation score for a given point."""
        path = self.path(point)
        isolation_node = path[self.node_sizes[path].argmin()]
        isolation_depth = self.depths[isolation_node]
        correction = self.expected_subtree_depth[isolation_node]
        return 2 ** -((isolation_depth + correction) / self.expected_subtree_depth[0])

    @classmethod
    @partial(jax.jit, static_argnames=("cls", "hyperplane_components"))
    def fit(cls, rng: jax.Array, data: jax.Array, hyperplane_components: int):
        """Fit a balanced (keep splitting on single point) isolation tree to the given data."""
        points, features = data.shape
        max_depth = np.ceil(np.log2(points)).astype(int)
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
        # it is faster in practice, especially on GPU due better memory patterns
        dot_prod_batch = jax.vmap(jax.vmap(jnp.dot, in_axes=(0, None)), in_axes=(None, 0))
        distances = dot_prod_batch(data, normals)

        # keep track of which nodes have been reached by each data point
        # splitting with a single point "forks" the path, so this needs to be
        # bool of shape (nodes, points) instead of int of shape (depth, points).
        # at the start, only the root node has been reached by all points
        reached = jnp.zeros((nodes, points), dtype=bool)
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
            intercepts_at_slice = jax.vmap(sample_intercept)(
                jax.random.split(rng_layer, 2**depth),
                distances[layer_slice],
                reached[layer_slice],
            )
            intercepts = intercepts.at[layer_slice].set(intercepts_at_slice)

            # update the reached masks for the children nodes
            reached = reached.at[2 * layer_slice + 1, :].set(
                reached[layer_slice] & (distances[layer_slice] >= intercepts[layer_slice, None])
            )
            reached = reached.at[2 * layer_slice + 2, :].set(
                reached[layer_slice] & (distances[layer_slice] <= intercepts[layer_slice, None])
            )
        node_sizes = reached.sum(axis=1)
        depths = jnp.log2(1 + jnp.arange(nodes)).astype(int)
        return cls(normals, intercepts, depths, node_sizes, expected_depth(node_sizes))


class ExtendedIsolationForest(struct.PyTreeNode):
    trees: IsolationTree

    @jax.jit
    def score(self, point: jax.Array) -> jax.Array:
        tree_scores = jax.vmap(IsolationTree.score, in_axes=(0, None))(self.trees, point)
        return jnp.exp(jnp.log(tree_scores).mean())

    @jax.jit
    def score_samples(self, data: jax.Array) -> jax.Array:
        return jax.vmap(self.score)(data)

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
        return cls(trees)
