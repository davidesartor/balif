import numpy as np
from typing import NamedTuple, Optional, Literal
from jaxtyping import Float, Int, Bool, Shaped
from scipy.special import digamma, comb

from bad import BetaDistr, BayesianDetector


def get_queries(
    model: BayesianDetector,
    X: Float[np.ndarray, "samples features"],
    batch_size: int = 1,
    mask: Optional[Bool[np.ndarray, "samples"]] = None,
) -> Int[np.ndarray, "batch"]:
    # initialize the mask if not provided
    if mask is None:
        mask = np.ones(X.shape[0], dtype=bool)
    mask = mask.copy()

    # pre-calculate the leaves entropy for each possible leaf-size (0 to batch_size)
    entropy_leaves = [
        np.zeros(model.beliefs.a.shape)
    ]  # 0_th elem never accessed, makes indexing easier
    for n in range(1, batch_size + 1):
        A = np.cumprod(model.beliefs.a[..., None] + np.arange(n), axis=-1)
        B = np.cumprod(model.beliefs.b[..., None] + np.arange(n), axis=-1)
        A = np.concatenate([np.ones_like(A[..., :1]), A], axis=-1)
        B = np.concatenate([np.ones_like(B[..., :1]), B], axis=-1)
        AB = np.prod(model.beliefs.ss()[..., None] + np.arange(n), axis=-1)
        P = A * B[..., ::-1] / AB[..., None]
        entropy_leaves.append(-np.sum(comb(n, np.arange(n + 1)) * P * np.log(P), axis=-1))
    entropy_leaves = np.stack(entropy_leaves, axis=0)  # (batch_size+1, estimators, regions)

    # gather the beliefs for each estimator and sample
    regions = model.estimators_apply(X)  # (estimators, samples)
    beliefs = model.gather_beliefs(regions)  # (estimators, samples)

    # gather the precomputed entropy for each estimator and sample
    entropy_leaves = np.take_along_axis(entropy_leaves, regions[None, ...], axis=-1)

    # pre-calculate the leaves condtional entropy for leaf-size 1
    a, b, mu = beliefs.a, beliefs.b, beliefs.mu()
    conditional_entropy_leaves = (
        digamma(a + b + 1) - mu * digamma(a + 1) - (1 - mu) * digamma(b + 1)
    )  # (estimators, samples)

    queries_idxs = []
    queries_in_regions = np.zeros(beliefs.a.shape, dtype=int)  # (estimators, samples)
    for i in range(batch_size):
        groups_entropy = np.take_along_axis(
            entropy_leaves, (queries_in_regions + 1)[None, ...], axis=0
        )  # (1, estimators, samples)
        groups_entropy = groups_entropy.squeeze(axis=0)  # (estimators, samples)
        groups_conditional_entropy = conditional_entropy_leaves * (queries_in_regions + 1)

        # query most interesting point in the worst case
        scores = (groups_entropy - groups_conditional_entropy).mean(axis=0)
        queries_idxs.append(np.where(mask, scores, -np.inf).argmax())

        # update the mask and regions onehot
        mask[queries_idxs[-1]] = False
        queries_in_regions += regions == regions[..., queries_idxs[-1]][..., None]
    return np.array(queries_idxs)
