import numpy as np
from typing import Optional, Literal
from jaxtyping import Float, Int, Bool, Shaped
from scipy.special import digamma, comb

from .bad import BetaDistr, BayesianDetector


def interest(
    beliefs: Shaped[BetaDistr, "samples"],
    method: Literal["margin", "anom", "bald"] = "margin",
) -> Float[np.ndarray, "samples"]:
    if method == "margin":
        a, b, r = beliefs.a, beliefs.b, 0.5
        mode = beliefs.mode().clip(0.01, 0.99)
        log_margin = (a - 1) * np.log(mode / r) + (b - 1) * np.log((1 - mode) / (1 - r))
        return np.exp(-log_margin)
    elif method == "anom":
        return beliefs.mu()
    elif method == "bald":
        a, b, mu = beliefs.a, beliefs.b, beliefs.mu()
        H_y = -mu * np.log(mu) - (1 - mu) * np.log(1 - mu)
        H_y_given_w = digamma(a + b + 1) - mu * digamma(a + 1) - (1 - mu) * digamma(b + 1)
        I_yw = H_y - H_y_given_w
        return I_yw
    else:
        raise ValueError(f"Unknown interest method: {self.interest_method}")


def get_queries_independent(
    model: BayesianDetector,
    X: Float[np.ndarray, "samples features"],
    interest_method: Literal["margin", "anom", "bald"] = "margin",
    batch_size: int = 1,
    mask: Optional[Bool[np.ndarray, "samples"]] = None,
) -> Int[np.ndarray, "batch"]:
    # gather aggregated beliefs for each sample
    regions = model.estimators_apply(X)
    beliefs = model.gather_beliefs(regions)
    beliefs = model.aggregate_beliefs(beliefs)

    # return top k samples
    scores = interest(beliefs, method=interest_method)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    queries_idxs = scores.argsort()[-batch_size:]
    return queries_idxs


def get_queries_worstcase(
    model: BayesianDetector,
    X: Float[np.ndarray, "samples features"],
    interest_method: Literal["margin", "anom", "bald"] = "margin",
    batch_size: int = 1,
    mask: Optional[Bool[np.ndarray, "samples"]] = None,
) -> Int[np.ndarray, "batch"]:
    # gather the beliefs for each estimator and sample
    regions = model.estimators_apply(X)
    beliefs = model.gather_beliefs(regions)

    queries_idxs = []
    queries_in_regions = np.zeros(beliefs.a.shape, dtype=int)
    mask = np.ones(len(X), dtype=bool) if mask is None else mask.copy()
    for i in range(batch_size):
        # get the worst case candidates
        most_anom = BetaDistr(a=beliefs.a + queries_in_regions, b=beliefs.b)
        lest_anom = BetaDistr(a=beliefs.a, b=beliefs.b + queries_in_regions)

        # aggregate the beliefs for each sample
        most_anom = model.aggregate_beliefs(most_anom)
        lest_anom = model.aggregate_beliefs(lest_anom)

        # compute interest score in the worst case
        scores_most_anom = interest(most_anom, method=interest_method)
        scores_lest_anom = interest(lest_anom, method=interest_method)
        scores = np.minimum(scores_most_anom, scores_lest_anom)
        scores = np.where(mask, scores, -np.inf)
        assert scores.shape == (len(X),)

        # query most interesting point in the worst case
        idx = scores.argmax()
        queries_idxs.append(idx)
        mask[idx] = False
        queries_in_regions += regions == regions[..., idx][..., None]
    return np.array(queries_idxs)


def get_queries_batchbald(
    model: BayesianDetector,
    X: Float[np.ndarray, "samples features"],
    batch_size: int = 1,
    mask: Optional[Bool[np.ndarray, "samples"]] = None,
) -> Int[np.ndarray, "batch"]:
    # pre-calculate the leaves entropy for each possible leaf-size (0 to batch_size)
    entropy_leaves = [np.zeros(model.beliefs.a.shape)]  # never used, makes indexing easier
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
    queries_in_regions = np.zeros(beliefs.a.shape, dtype=int)
    mask = np.ones(len(X), dtype=bool) if mask is None else mask.copy()
    for i in range(batch_size):
        groups_entropy = np.take_along_axis(
            entropy_leaves, (queries_in_regions + 1)[None, ...], axis=0
        ).squeeze(axis=0)
        groups_conditional_entropy = conditional_entropy_leaves * (queries_in_regions + 1)

        # query most interesting point in the worst case
        scores = (groups_entropy - groups_conditional_entropy).mean(axis=0)
        scores = np.where(mask, scores, -np.inf)

        # query most interesting point in the worst case
        idx = scores.argmax()
        queries_idxs.append(idx)
        mask[idx] = False
        queries_in_regions += regions == regions[..., idx][..., None]
    return np.array(queries_idxs)
