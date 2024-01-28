import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

import matplotlib.pyplot as plt
from typing import Any, Callable, Optional, NamedTuple
from functools import partial


def sample_vector(rng: jax.Array, features: jnp.int8, components: jnp.int8):
    rng_value, rng_idx = jax.random.split(rng)
    if components == 1:
        idx = jax.random.randint(rng_idx, (), 0, features)
        return jnp.zeros((features,)).at[idx].set(1.0)

    idxs = jax.random.choice(rng_idx, features, (components,), replace=False)
    values = jax.random.normal(rng_value, shape=(components,))
    vector = jnp.zeros((features,)).at[idxs].set(values)
    return vector / jnp.linalg.norm(vector)


def sample_intercept(rng: jax.Array, distances: jax.Array, mask: jax.Array):
    min_distance = jnp.where(mask, distances, jnp.inf).min()
    max_distance = jnp.where(mask, distances, -jnp.inf).max()
    intercept = jax.random.uniform(rng, (), minval=min_distance, maxval=max_distance)
    return intercept


class TreeNodes(NamedTuple):
    node_sizes: jax.Array
    normals: jax.Array
    intercepts: jax.Array

    @property
    def max_depth(self) -> jnp.int8:
        return jnp.log2(self.node_sizes.shape[0] + 1).astype(np.int8) - 1


def distance(normal: jax.Array, intercept: jax.Array, point: jax.Array):
    return jnp.dot(normal, point) - intercept


def make_tree(rng: jax.Array, data: jax.Array, hyperplane_components: jnp.int8) -> TreeNodes:
    """Constructs a binary tree for data partitioning using random hyperplanes.

    The data points that reach each node are partitioned into its two children nodes.
    The partitioning is done by sampling a random hyperplane that cuts through the data.
    The resulting tree is a balanced bynary tree of depth floor(log2(points)).
    If a single point reaches a node, the subsequent splits are random hyperplanes
    passing through that point. Isolation points are not the leaves of the tree.

    Args:
        rng (jax.Array): Key for the random number generator.
        data (jax.Array): Input data of shape (points, features).
        hyperplane_components (jnp.int8): Number of non-zero components in the hyperplane.

    Returns:
        TreeNodes: A named tuple with the following fields:
            n_points (jax.Array): Number of points that reach each node.
            normals (jax.Array): Normal vectors of the hyperplanes.
            intercepts (jax.Array): Intercept of the hyperplanes.
    """
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
    return TreeNodes(reached.sum(axis=1), normals, intercepts)


def path(point: jax.Array, tree: TreeNodes):
    def scan_body(id, x):
        child = jax.lax.select(
            distance(tree.normals[id], tree.intercepts[id], point) >= 0,
            2 * id + 1,
            2 * id + 2,
        )
        return child, id

    leaf, path = jax.lax.scan(
        scan_body, jnp.zeros((), dtype=jnp.int8), None, length=tree.max_depth - 1
    )
    return jnp.concatenate((path, leaf[None,]))
