from typing import Literal, Optional, Protocol, NamedTuple, Self
from jaxtyping import Array, Float, Bool, Int, PRNGKeyArray
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np


class IsolationTree(eqx.Module):
    fit_data: Float[Array, "samples dim"]
    reached: Int[Array, "nodes+leaves"]
    normals: Float[Array, "nodes dim"]
    intercepts: Float[Array, "nodes"]

    @classmethod
    def fit(cls, data: Float[Array, "samples dim"], *, key: PRNGKeyArray, **kwargs):
        samples, dim = data.shape
        max_depth = np.ceil(np.log2(samples)).astype(int)
        nodes, leaves = 2**max_depth - 1, 2**max_depth

        @jax.vmap
        def sample_hyperplanes(reached, key):
            return cls.sample_hyperplane_split(data, reached, key=key, **kwargs)

        normals, intercepts = jnp.empty((nodes, dim)), jnp.empty((nodes,))
        reached = jnp.zeros((nodes + leaves, samples), dtype=bool).at[0, :].set(True)

        for depth, key_layer in enumerate(jr.split(key, max_depth)):
            layer = jnp.arange(2**depth - 1, 2**depth - 1 + 2**depth)
            normals_at_layer, intercepts_at_layer, distances = sample_hyperplanes(
                reached[layer], jr.split(key_layer, 2**depth)
            )
            normals = normals.at[layer].set(normals_at_layer)
            intercepts = intercepts.at[layer].set(intercepts_at_layer)

            reached_left = jnp.logical_and(reached[layer], distances >= 0)
            reached_right = jnp.logical_and(reached[layer], distances <= 0)
            reached = reached.at[2 * layer + 1, :].set(reached_left)
            reached = reached.at[2 * layer + 2, :].set(reached_right)

        return cls(data, reached.sum(axis=-1), normals, intercepts)

    @staticmethod
    def sample_hyperplane_split(
        data: Float[Array, "samples dim"],
        mask: Bool[Array, "samples"],
        *,
        key: PRNGKeyArray,
        hyperplane_components: int,
        p_normal_idx: Literal["uniform", "range"],
        p_normal_value: Literal["uniform", "range", "covariant"],
        p_intercept: Literal["uniform", "normal"],
    ):
        samples, dim = data.shape
        if hyperplane_components <= 0:
            hyperplane_components = dim
        min_data = jnp.min(data, axis=0, initial=jnp.inf, where=mask[:, None])
        max_data = jnp.max(data, axis=0, initial=-jnp.inf, where=mask[:, None])

        def sample_normal_idx(key):
            if p_normal_idx == "range":
                p = max_data - min_data
            elif p_normal_idx == "uniform":
                p = None
            else:
                raise ValueError(f"Unknown normal_idx distribution {p_normal_idx}")
            idxs = jr.choice(key, dim, (hyperplane_components,), replace=False, p=p)
            return idxs

        def sample_normal_values(key, idx):
            if p_normal_value == "uniform":
                values = jr.normal(key, (hyperplane_components,))
            elif p_normal_value == "range":
                mean = jnp.zeros((hyperplane_components,))
                cov = jnp.diag(max_data[idx] - min_data[idx])
                cov = cov + 1e-8 * jnp.eye(hyperplane_components)
                values = jr.multivariate_normal(key, mean, cov, method="svd")
            elif p_normal_value == "covariant":
                mean = jnp.zeros((hyperplane_components,))
                cov = jnp.cov(data.T, fweights=mask.astype(int))
                cov = jnp.nan_to_num(cov[idx][:, idx], nan=0.0)
                cov = cov + 1e-8 * jnp.eye(hyperplane_components)
                values = jr.multivariate_normal(key, mean, cov, method="svd")
            else:
                raise ValueError(f"Unknown normal_value distribution {p_normal_value}")
            normal = jnp.zeros((dim,)).at[idx].set(values)
            distances = jnp.dot(data, normal)
            scale = distances.std(where=mask)
            scale = jax.lax.select(scale == 0, 1.0, scale)
            return normal / scale, distances / scale

        def sample_intercept(key, distances):
            min_dist = jnp.min(distances, initial=jnp.inf, where=mask)
            max_dist = jnp.max(distances, initial=-jnp.inf, where=mask)
            if p_intercept == "normal":
                mean, std = (max_dist - min_dist) / 2, (max_dist - min_dist) / 6
                intercept = mean + std * jr.normal(key)
                intercept = jnp.clip(intercept, min_dist, max_dist)
            elif p_intercept == "uniform":
                intercept = jr.uniform(key, minval=min_dist, maxval=max_dist)
            else:
                raise ValueError(f"Unknown intercept distribution {p_intercept}")
            distances = distances - intercept
            return intercept, distances

        key_idx, key_value, key_intercept = jr.split(key, 3)
        normal_idx = sample_normal_idx(key_idx)
        normal, distances = sample_normal_values(key_value, normal_idx)
        intercept, distances = sample_intercept(key_intercept, distances)
        return normal, intercept, distances

    @staticmethod
    def depth(node_id: Int[Array, "..."]) -> Int[Array, "..."]:
        return jnp.log2(1 + node_id).astype(int)

    @staticmethod
    def expected_depth(node_size: Int[Array, "..."]) -> Float[Array, "..."]:
        """Compute the expected isolation depth for a node with n data points."""
        EULER_MASCHERONI = 0.5772156649
        harmonic_number = jax.lax.select(
            node_size > 1, jnp.log(node_size - 1) + EULER_MASCHERONI, 0.0
        )
        expected_depth = 2 * harmonic_number - 2 * (node_size - 1) / node_size
        return expected_depth

    def child(
        self, node: Int[Array, ""], data: Float[Array, "dim"], *, key: PRNGKeyArray
    ) -> Bool[Array, ""]:
        normal, intercept = self.normals[node], self.intercepts[node]
        distance = jnp.dot(data, normal) - intercept + 1e-8 * jr.normal(key)
        return jax.lax.select(distance > 0, 2 * node + 1, 2 * node + 2)

    def path(
        self, data: Float[Array, "dim"], *, key: PRNGKeyArray
    ) -> tuple[Int[Array, ""], Int[Array, "max_depth"]]:
        def scan_body(node, key):
            stop_cond = self.reached[node] <= 1
            next = jax.lax.select(stop_cond, node, self.child(node, data, key=key))
            return next, node

        root = jnp.zeros((), int)
        max_depth = int(np.log2(self.reached.shape[-1] + 1))
        terminal, path = jax.lax.scan(scan_body, root, jr.split(key, max_depth - 1))
        return terminal, path


class IsolationForest(eqx.Module):
    n_estimators: int = eqx.field(default=128, static=True)
    max_samples: int = eqx.field(default=256, static=True)
    bootstrap: bool = eqx.field(default=True, static=True)
    standardize: bool = eqx.field(default=False, static=True)

    hyperplane_components: int = eqx.field(default=1, static=True)
    p_normal_idx: Literal["uniform", "range", "covariant"] = eqx.field(
        default="uniform", static=True
    )
    p_normal_value: Literal["uniform", "range", "covariant"] = eqx.field(
        default="uniform", static=True
    )
    p_intercept: Literal["uniform", "normal"] = eqx.field(
        default="uniform", static=True
    )

    trees: IsolationTree = eqx.field(
        default_factory=lambda: IsolationTree.fit(jnp.empty((1, 1)), key=jr.key(0))
    )

    @eqx.filter_jit
    def fit(self, data: Float[Array, "samples dim"], *, key: PRNGKeyArray) -> Self:
        max_samples = min(self.max_samples, data.shape[0])
        if self.standardize:
            std = data.std(axis=0)
            data = (data - data.mean(axis=0)) / jnp.where(std > 0, std, 1.0)

        def fit_tree(key):
            key_sample, key_fit = jr.split(key)
            return IsolationTree.fit(
                data=jr.choice(
                    key_sample, data, (max_samples,), replace=self.bootstrap
                ),
                key=key_fit,
                hyperplane_components=self.hyperplane_components,
                p_normal_idx=self.p_normal_idx,
                p_normal_value=self.p_normal_value,
                p_intercept=self.p_intercept,
            )

        trees = jax.vmap(fit_tree)(jr.split(key, self.n_estimators))
        return eqx.tree_at(lambda x: x.trees, self, trees)

    @eqx.filter_jit
    def score(
        self, data: Float[Array, "*batch dim"], *, key: PRNGKeyArray
    ) -> Float[Array, "*batch"]:
        def score_fn(tree, point, key) -> Float[Array, ""]:
            isolation_node, _ = tree.path(point, key=key)
            correction = tree.expected_depth(tree.reached[isolation_node])
            normalization = tree.expected_depth(tree.reached[0])
            score = 2 ** (-(tree.depth(isolation_node) + correction) / normalization)
            return score

        def ensamble_score_fn(point):
            keys = jr.split(key, self.n_estimators)
            scores = jax.vmap(score_fn, in_axes=(0, None, 0))(self.trees, point, keys)
            return jnp.exp(jnp.mean(jnp.log(scores)))

        *batch, dim = data.shape
        for _ in batch:
            ensamble_score_fn = jax.vmap(ensamble_score_fn)
        return ensamble_score_fn(data)
