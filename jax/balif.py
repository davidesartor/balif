from functools import partial
from typing import NamedTuple, Optional
import jax
import jax.numpy as jnp
from flax import struct
from isolation_forest import IsolationTree, ExtendedIsolationForest


class BetaDistribution(NamedTuple):
    alpha: jnp.float_
    beta: jnp.float_

    @classmethod
    def from_mean_and_sample_size(cls, mean: jnp.float_, sample_size: jnp.float_):
        return cls(mean * sample_size, (1 - mean) * sample_size)

    @property
    def mean(self) -> jnp.float_:
        return self.alpha / (self.alpha + self.beta)

    @property
    def mode(self) -> jnp.float_:
        return jax.lax.select(
            jnp.minimum(self.alpha, self.beta) > 1,
            (self.alpha - 1) / (self.alpha + self.beta - 2),
            jax.lax.select(self.alpha > self.beta, 1.0, 0.0),
        )

    def logpdf(self, x: jnp.float_) -> jnp.float_:
        return jax.scipy.stats.beta.logpdf(x, self.alpha, self.beta)


class BalifTree(IsolationTree):
    alphas: jax.Array  # shape (nodes,)
    betas: jax.Array  # shape (nodes,)

    @classmethod
    @partial(jax.jit, static_argnames=("cls",))
    def from_isolation_tree(cls, itree: IsolationTree, prior_sample_size: jnp.float_):
        def base_scores(itree: IsolationTree) -> jax.Array:
            """Compute the IF score for each node in the tree"""
            n_nodes, n_features = itree.normals.shape

            # single tree IF score. Note that this is not a good way to set the prior
            # the root has score of 0.5 -> contamination = 0.5
            # a node separated at first step has score of 2**(1/(2log(psi)-1)) ~= 0.95
            # a node at maxdepth with all points has score of 2**-1 -log2(psi)/(2log(psi)-1) ~= 0.3
            node_depths = jax.vmap(itree.depth)(jnp.arange(n_nodes))
            depth_corrections = jax.vmap(itree.expected_depth)(itree.node_sizes)
            expected_from_root = itree.expected_depth(itree.node_sizes[0])
            normalized_depths = (node_depths + depth_corrections) / expected_from_root
            base_scores = 2**-normalized_depths
            return base_scores

        def get_priors(itree: IsolationTree):
            """Compute the prior for each node in the tree"""
            # rescale the scores to be in the range [0, 1]
            scores = (4 / 3) * (base_scores(itree) - 0.25)

            # clips prediction for "phantom" nodes after isolation (sets it to the most extreme value in the tree)
            # scores = jnp.where(itree.node_sizes > 1, scores, jnp.max(scores))

            # match the predition adding strictly positive virtual samples
            sample_size_after_IF = prior_sample_size / jnp.minimum(scores, 1 - scores)
            alphas = sample_size_after_IF * scores
            betas = sample_size_after_IF * (1 - scores)
            return alphas, betas

        alphas, betas = get_priors(itree)
        return cls(itree.normals, itree.intercepts, itree.node_sizes, alphas, betas)


class Balif(struct.PyTreeNode):
    trees: BalifTree

    @jax.jit
    def prediction_as_distr(self, point: jax.Array) -> BetaDistribution:
        def tree_prediction_as_distr(tree) -> BetaDistribution:
            isolation_node = tree.isolation_node(point)
            alpha = tree.alphas[isolation_node]
            beta = tree.betas[isolation_node]
            return BetaDistribution(alpha, beta)

        alphas, betas = jax.vmap(tree_prediction_as_distr)(self.trees)
        combined_mean = jnp.mean(alphas / (alphas + betas), axis=0)
        combined_sample_size = alphas.sum(axis=0) + betas.sum(axis=0)
        return BetaDistribution.from_mean_and_sample_size(combined_mean, combined_sample_size)

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
        return jax.vmap(self.register)(data, are_anomaly)

    @jax.jit
    def interest_for(self, data: jax.Array) -> jax.Array:
        def interest_for_point(point: jax.Array) -> jax.Array:
            distr = self.prediction_as_distr(point)
            logmargin = distr.logpdf(distr.mode) - distr.logpdf(x=0.5)
            return jnp.exp(-logmargin)

        return jax.vmap(interest_for_point)(data)

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
        return jnp.array(queries_idx)

    @classmethod
    @partial(
        jax.jit, static_argnames=("cls", "n_estimators", "max_samples", "hyperplane_components")
    )
    def fit(cls, rng: jax.Array, data: jax.Array, *, prior_sample_size=0.1, **kwargs):
        batch_fit_from_itree = jax.vmap(BalifTree.from_isolation_tree, in_axes=(0, None))
        itrees = ExtendedIsolationForest.fit(rng, data, **kwargs).trees
        return cls(trees=batch_fit_from_itree(itrees, prior_sample_size))
