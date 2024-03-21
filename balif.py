from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from flax import struct
import numpy as np
from isolation_forest import IsolationTree, ExtendedIsolationForest


class BetaDistribution(NamedTuple):
    alpha: jax.Array
    beta: jax.Array

    @classmethod
    def from_mean_and_sample_size(cls, mean: jax.Array, sample_size: jax.Array):
        return cls(mean * sample_size, (1 - mean) * sample_size)

    @classmethod
    def from_mean_and_variance(cls, mean: jax.Array, variance: jax.Array):
        alpha = mean * (mean * (1 - mean) / variance - 1)
        beta = (1 - mean) * (mean * (1 - mean) / variance - 1)
        return cls(alpha, beta)

    @property
    def samplesize(self) -> jax.Array:
        return self.alpha + self.beta

    @property
    def mean(self) -> jax.Array:
        return self.alpha / self.samplesize

    @property
    def mode(self) -> jax.Array:
        return jax.lax.select(
            jnp.minimum(self.alpha, self.beta) > 1,
            (self.alpha - 1) / (self.samplesize - 2),
            jax.lax.select((self.alpha > self.beta), 1.0, 0.0),
        )

    @property
    def variance(self) -> jax.Array:
        return self.alpha * self.beta / ((self.samplesize + 1) * (self.samplesize) ** 2)

    def logpdf(self, x: jnp.float_) -> jnp.float_:
        return jax.scipy.stats.beta.logpdf(x, self.alpha, self.beta)


def get_score_matrix(max_depth, max_samples):
    def p(start, end):
        start, end = start - 2, end - 1
        bin_coef_ln = gammaln(start + 1) - gammaln(end + 1) - gammaln(start - end + 1)
        return jnp.exp(bin_coef_ln + start * jnp.log(0.5))

    # get the transition matrix for the Markov chain of splitting sizes
    M = jax.vmap(jax.vmap(p, in_axes=(0, None)), in_axes=(None, 0))(
        jnp.arange(1, max_samples + 1), jnp.arange(1, max_samples + 1)
    )
    M = M.at[:, 0].set(0).at[0, 0].set(1)

    # start with a prob vector with all mass in the last element (max node size)
    p = jnp.zeros(max_samples).at[-1].set(1)

    # compute the k-step transition matrix for k = 0, ..., hmax
    Mk = jax.lax.associative_scan(jnp.matmul, jnp.stack([M] * max_depth))
    Mk = jnp.concatenate([jnp.eye(max_samples)[jnp.newaxis], Mk], axis=0)

    # compute p after i steps for i = 0, ..., hmax
    # add zero so indexing is correct
    p = jax.vmap(jnp.matmul, in_axes=(0, None))(Mk, p)
    p = jnp.concatenate([jnp.zeros((max_depth + 1, 1)), p], axis=1)

    # compute the cumulative distribution function
    pc = p.cumsum(axis=1)
    pc = (pc.at[:, 1:].add(pc[:, :-1])) / 2  # assume points are "middle of the pack"
    pc = jnp.clip(pc, 0.01, 0.99).at[:, 0].set(0)  # fix floating point errors
    return pc


class BalifTree(IsolationTree):
    alphas: jax.Array  # shape (nodes,)
    betas: jax.Array  # shape (nodes,)

    @classmethod
    @partial(jax.jit, static_argnames=("cls"))
    def from_isolation_tree(
        cls,
        itree: IsolationTree,
        score_matrix: jax.Array,
        prior_sample_size: jnp.float_,
    ):
        def base_scores(itree: IsolationTree) -> jax.Array:
            def scan_score(_, idx):
                return _, 1 - score_matrix[idx[0], idx[1]]

            n_nodes, n_features = itree.normals.shape
            node_depths = jax.vmap(itree.depth)(jnp.arange(n_nodes))
            _, scores = jax.lax.scan(scan_score, None, (node_depths, itree.node_sizes))
            return scores

        def get_priors(itree: IsolationTree):
            """Compute the prior for each node in the tree"""
            scores = base_scores(itree)

            # match the predition adding strictly positive virtual samples
            sample_size_after_IF = (
                prior_sample_size  # / (jnp.minimum(scores, 1 - scores))
            )
            alphas = sample_size_after_IF * scores
            betas = sample_size_after_IF * (1 - scores)
            return alphas, betas

        alphas, betas = get_priors(itree)
        return cls(itree.normals, itree.intercepts, itree.node_sizes, alphas, betas)


class Balif(struct.PyTreeNode):
    trees: BalifTree
    path_score: bool

    @jax.jit
    def prediction_as_distr(self, point: jax.Array) -> BetaDistribution:
        def path_score(tree):
            path = tree.path(point)
            isolation_node = path[tree.node_sizes[path].argmin()]
            weights = jnp.where(path <= isolation_node, 2 ** jnp.arange(len(path)), 0)
            # weights = 2 ** jnp.arange(len(path))
            alpha = jnp.mean(tree.alphas[path] * weights / weights.sum())
            beta = jnp.mean(tree.betas[path] * weights / weights.sum())
            return BetaDistribution(alpha, beta)

        def isolation_score(tree) -> BetaDistribution:
            isolation_node = tree.isolation_node(point)
            alpha, beta = tree.alphas[isolation_node], tree.betas[isolation_node]
            return BetaDistribution(alpha, beta)

        distributions = jax.lax.cond(
            self.path_score,
            jax.vmap(path_score),
            jax.vmap(isolation_score),
            operand=self.trees,
        )
        # combined_mean = 1 - jnp.exp(jnp.log(betas / (alphas + betas)).mean(axis=0))

        return BetaDistribution.from_mean_and_variance(
            mean=jnp.mean(distributions.mean, axis=-1),
            variance=jnp.mean(distributions.variance, axis=-1) / len(distributions),
        )

    @jax.jit
    def score(self, point: jax.Array) -> jax.Array:
        return self.prediction_as_distr(point).mean

    @jax.jit
    def score_samples(self, data: jax.Array) -> jax.Array:
        return jax.vmap(self.score)(data)

    @jax.jit
    def register(self, point: jax.Array, is_anomaly: bool):
        def update_tree(tree):
            path = tree.path(point)
            new_alphas = tree.alphas.at[path].add(jax.lax.select(is_anomaly, 1, 0))
            new_betas = tree.betas.at[path].add(jax.lax.select(is_anomaly, 0, 1))
            return tree.replace(alphas=new_alphas, betas=new_betas)

        return self.replace(trees=jax.vmap(update_tree)(self.trees))

    @jax.jit
    def register_samples(self, data: jax.Array, are_anomaly: jax.Array):
        def update(model, query):
            point, is_anomaly = query
            return model.register(point, is_anomaly), None

        updated_model, _ = jax.lax.scan(update, self, (data, are_anomaly))
        return updated_model

    @jax.jit
    def interest_for(self, data: jax.Array, r=0.5) -> jax.Array:
        def interest_for_point(point: jax.Array) -> jax.Array:
            distr = self.prediction_as_distr(point)
            logmargin = distr.logpdf(distr.mode) - distr.logpdf(x=r)
            return jnp.exp(-logmargin)

        return jax.vmap(interest_for_point)(data)

    @partial(jax.jit, static_argnames=("k",))
    def get_batch_queries(self, data: jax.Array, k: int = 1):
        def merge_superpositions(model_superpos1, model_superpos2):
            alphas1, betas1 = model_superpos1.trees.alphas, model_superpos1.trees.betas
            alphas2, betas2 = model_superpos2.trees.alphas, model_superpos2.trees.betas
            trees = model_superpos1.trees.replace(
                alphas=jnp.concatenate([alphas1, alphas2], axis=-1),
                betas=jnp.concatenate([betas1, betas2], axis=-1),
            )
            return model_superpos1.replace(trees=trees)

        queries_idx = []
        model_superpos = self.replace(
            trees=self.trees.replace(
                alphas=self.trees.alphas[..., jnp.newaxis],
                betas=self.trees.betas[..., jnp.newaxis],
            )
        )

        for _ in range(k):
            interests_worst_case = model_superpos.interest_for(data).min(axis=-1)
            query_idx = interests_worst_case.argmax()
            queries_idx.append(query_idx)

            model_superpos = merge_superpositions(
                model_superpos.register(data[query_idx], is_anomaly=True),
                model_superpos.register(data[query_idx], is_anomaly=False),
            )
        return jax.Array(queries_idx)

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
        max_samples: Optional[int] = None,
        prior_sample_size=0.1,
        path_score=False,
        **kwargs,
    ):
        max_samples = max_samples or min((256, data.shape[0]))
        batch_fit_from_itree = jax.vmap(
            BalifTree.from_isolation_tree, in_axes=(0, None, None)
        )
        itrees = ExtendedIsolationForest.fit(
            rng, data, max_samples=max_samples, **kwargs
        ).trees

        max_depth = np.log2(max_samples).astype(int)
        score_matrix = get_score_matrix(max_depth, max_samples)
        return cls(
            trees=batch_fit_from_itree(itrees, score_matrix, prior_sample_size),
            path_score=path_score,
        )
