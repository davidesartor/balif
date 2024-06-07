from functools import partial
from typing import Optional
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np


class IsolationTree(struct.PyTreeNode):
    normals: jax.Array  # shape (nodes, features)
    intercepts: jax.Array  # shape (nodes,)
    node_sizes: jax.Array  # shape (nodes,)

    @staticmethod
    def depth(node_id: jnp.int_) -> jnp.int_:
        return jnp.log2(1 + node_id).astype(int)

    @staticmethod
    def expected_depth(node_size: jnp.int_) -> jnp.float_:
        """Compute the expected isolation depth for a node with n data points."""
        EULER_MASCHERONI = 0.5772156649
        harm_number = jax.lax.select(
            node_size > 1, jnp.log(node_size - 1) + EULER_MASCHERONI, 0.0
        )
        expected_depth = 2 * harm_number - 2 * (node_size - 1) / node_size
        return expected_depth

    @jax.jit
    def path(self, point: jax.Array) -> jax.Array:
        """Return the path for a given point in the tree as an Array of node idx."""
        n_nodes, n_features = self.normals.shape

        def scan_body(node_id, x):
            distance = jnp.dot(self.normals[node_id], point) - self.intercepts[node_id]
            child = jax.lax.select(distance >= 0, 2 * node_id + 1, 2 * node_id + 2)
            return child, node_id

        max_depth = int(np.log2(n_nodes + 1))
        _, path = jax.lax.scan(scan_body, 0, None, length=max_depth)
        return path

    @jax.jit
    def isolation_node(self, point: jax.Array) -> jnp.int_:
        """Return the leaf for a given point in the tree as an Array of node idx."""
        n_nodes, n_features = self.normals.shape

        def while_cond(node_id):
            return (2 * node_id + 2 < n_nodes) & (self.node_sizes[node_id] > 1)

        def while_body(node_id: jnp.int_) -> jnp.int_:
            distance = jnp.dot(self.normals[node_id], point) - self.intercepts[node_id]
            child_id = jax.lax.select(distance >= 0, 2 * node_id + 1, 2 * node_id + 2)
            return child_id

        isolation_node = jax.lax.while_loop(while_cond, while_body, 0)
        return isolation_node

    @classmethod
    @partial(jax.jit, static_argnames=("cls", "hyperplane_components"))
    def fit(cls, rng: jax.Array, data: jax.Array, hyperplane_components: int):
        """Fit a balanced (keep splitting on single point) isolation tree to the given data."""
        points, features = data.shape
        max_depth = np.ceil(np.log2(points)).astype(int)
        nodes = 2 ** (max_depth + 1) - 1
        rng_normal, rng_intercept = jax.random.split(rng)

        # for each node, initialize intercepts to 0.0 and presample normal vector
        def sample_normal(rng: jax.Array) -> jax.Array:
            rng_value, rng_idx = jax.random.split(rng)
            idxs = jax.random.choice(
                rng_idx, features, (hyperplane_components,), replace=False
            )
            values = jax.random.normal(rng_value, shape=(hyperplane_components,))
            vector = jnp.zeros((features,)).at[idxs].set(values)
            return vector / jnp.linalg.norm(vector)

        intercepts = jnp.zeros((nodes,))
        normals = jax.vmap(sample_normal)(jax.random.split(rng_normal, nodes))

        # precompute dot products between data points and normals
        # doing it this way technically requires more compute but
        # it is faster in practice, especially on GPU due better memory patterns
        dot_prod_batch = jax.vmap(
            jax.vmap(jnp.dot, in_axes=(0, None)), in_axes=(None, 0)
        )
        distances = dot_prod_batch(data, normals)  # shape (nodes, points)

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
            def sample_intercept(rng: jax.Array, distances: jax.Array, mask: jax.Array):
                min_distance = jnp.where(mask, distances, jnp.inf).min()
                max_distance = jnp.where(mask, distances, -jnp.inf).max()
                intercept = jax.random.uniform(
                    rng, (), minval=min_distance, maxval=max_distance
                )
                return intercept

            intercepts_at_slice = jax.vmap(sample_intercept)(
                jax.random.split(rng_layer, 2**depth),
                distances[layer_slice],
                reached[layer_slice],
            )
            intercepts = intercepts.at[layer_slice].set(intercepts_at_slice)

            # update the reached masks for the children nodes
            reached = reached.at[2 * layer_slice + 1, :].set(
                reached[layer_slice]
                & (distances[layer_slice] >= intercepts[layer_slice, None])
            )
            reached = reached.at[2 * layer_slice + 2, :].set(
                reached[layer_slice]
                & (distances[layer_slice] <= intercepts[layer_slice, None])
            )

        node_sizes = reached.sum(axis=1)
        return cls(normals, intercepts, node_sizes)


class BalancedIsolationTree(IsolationTree):
    @classmethod
    @partial(jax.jit, static_argnames=("cls", "hyperplane_components"))
    def fit(cls, rng: jax.Array, data: jax.Array, hyperplane_components: int):
        """Fit a balanced (keep splitting on single point) isolation tree to the given data."""
        points, features = data.shape
        max_depth = np.ceil(np.log2(points)).astype(int)
        nodes = 2 ** (max_depth + 1) - 1
        rng_normal, rng_intercept = jax.random.split(rng)

        # for each node, initialize intercepts to 0.0 and presample normal vector
        def sample_normal(rng: jax.Array) -> jax.Array:
            rng_value, rng_idx = jax.random.split(rng)
            idxs = jax.random.choice(
                rng_idx, features, (hyperplane_components,), replace=False
            )
            values = jax.random.normal(rng_value, shape=(hyperplane_components,))
            vector = jnp.zeros((features,)).at[idxs].set(values)
            return vector / jnp.linalg.norm(vector)

        # update the intercepts to make the plane split the data points
        def sample_intercept(rng: jax.Array, distances: jax.Array, mask: jax.Array):
            min_distance = jnp.where(mask, distances, jnp.inf).min()
            max_distance = jnp.where(mask, distances, -jnp.inf).max()
            intercept = jax.random.uniform(
                rng, (), minval=min_distance, maxval=max_distance
            )
            return intercept

        intercepts = jnp.zeros((nodes,))
        normals = jax.vmap(sample_normal)(jax.random.split(rng_normal, nodes))

        # precompute dot products between data points and normals
        # doing it this way technically requires more compute but
        # it is faster in practice, especially on GPU due better memory patterns
        dot_prod_batch = jax.vmap(
            jax.vmap(jnp.dot, in_axes=(0, None)), in_axes=(None, 0)
        )
        distances = dot_prod_batch(data, normals)  # shape (nodes, points)

        reached = jnp.zeros((nodes, points), dtype=bool)
        reached = reached.at[0, :].set(True)

        for depth, rng_layer in enumerate(jax.random.split(rng_intercept, max_depth)):
            layer_slice = jnp.arange(2**depth - 1, 2**depth - 1 + 2**depth)

            intercepts_at_slice = jax.vmap(sample_intercept)(
                jax.random.split(rng_layer, 2**depth),
                distances[layer_slice],
                reached[layer_slice],
            )
            intercepts = intercepts.at[layer_slice].set(intercepts_at_slice)

            # update the reached masks for the children nodes
            reached = reached.at[2 * layer_slice + 1, :].set(
                reached[layer_slice]
                & (distances[layer_slice] >= intercepts[layer_slice, None])
            )
            reached = reached.at[2 * layer_slice + 2, :].set(
                reached[layer_slice]
                & (distances[layer_slice] <= intercepts[layer_slice, None])
            )

        node_sizes = reached.sum(axis=1)
        return cls(normals, intercepts, node_sizes)


class ExtendedIsolationForest(struct.PyTreeNode):
    trees: IsolationTree

    @jax.jit
    def score(self, point: jax.Array) -> jax.Array:
        def normalized_depth(tree, point) -> jnp.float_:
            isolation_node = tree.isolation_node(point)
            isolation_depth = tree.depth(isolation_node)
            correction = tree.expected_depth(tree.node_sizes[isolation_node])
            return (isolation_depth + correction) / tree.expected_depth(
                tree.node_sizes[0]
            )

        normalized_depths = jax.vmap(normalized_depth, in_axes=(0, None))(
            self.trees, point
        )
        return 2 ** -normalized_depths.mean()

    @jax.jit
    def score_samples(self, data: jax.Array) -> jax.Array:
        return jax.vmap(self.score)(data)

    @classmethod
    @partial(
        jax.jit,
        static_argnames=("cls", "n_estimators", "max_samples", "hyperplane_components"),
    )
    def fit(
        cls,
        rng: jax.Array,
        data: jax.Array,
        *,
        n_estimators: int = 128,
        max_samples: Optional[int] = None,
        hyperplane_components: Optional[int] = None,
        bootstrap: bool = True,
    ):
        max_samples = max_samples or min((256, data.shape[0]))
        hyperplane_components = hyperplane_components or data.shape[-1]

        def fit_tree(rng):
            rng_sample, rng_fit = jax.random.split(rng)
            subsample = jax.random.choice(
                rng_sample, data, (max_samples,), replace=bootstrap
            )
            return IsolationTree.fit(rng_fit, subsample, hyperplane_components)

        trees = jax.vmap(fit_tree)(jax.random.split(rng, n_estimators))
        return cls(trees)
