"""
This script implements BALD and BatchBALD for active learning 
using the exact formula for conditional and unconditional entropies
in the case of BALIF.
"""
import numpy as np
from scipy.special import digamma
from typing import NamedTuple, List

class CandidateBatch(NamedTuple):
    scores: np.ndarray
    indices: np.ndarray


def compute_conditional_entropy(alphas, betas): 
    """
    Compute E_w[H(y|x,w)] assuming w~Beta(alpha, beta) and y|w~Bernoulli(w).
    E_w[H(y|x,w)] = - [alpha/(alpha+beta) * digamma(alpha+1) + \
                       beta/(alpha+beta)*digamma(beta+1)] + digamma(alpha+beta+1)

    Parameters
    ----------
    alphas : numpy.ndarray
        (n_samples, )
    betas : numpy.ndarray
        (n_samples, )
    
    Returns 
    -------
    numpy.ndarray
        (n_samples, )
    """
    sample_sizes = alphas + betas               
    means = alphas / sample_sizes               
    digammas_a1 = digamma(alphas + 1)            
    digammas_b1 = digamma(betas + 1)
    digammas_total1 = digamma(sample_sizes + 1)
    expected_entropy = - (means*digammas_a1+ (1-means)*digammas_b1) + digammas_total1
    return expected_entropy

def compute_entropy(alphas, betas): 
    """
    Compute H(y|x) assuming y|x,w~Bernoulli(w) and w~Beta(alpha, beta).
    H(y|x) = - mean * log2(mean) - (1-mean) * log2(1-mean) where mean = alpha/(alpha+beta).

    Parameters
    ----------
    alphas : numpy.ndarray
        (n_samples, )
    betas : numpy.ndarray
        (n_samples, )
    
    Returns 
    -------
    numpy.ndarray
        (n_samples, )
    """
    sample_sizes = alphas + betas
    means = alphas / sample_sizes
    entropies = - (means*np.log(means) + (1-means)*np.log(1-means))
    return entropies


def get_exactbald_batch(alphas, betas, batch_size:int=1): 
    N = alphas.shape[0]
    batch_size = min(batch_size, N)

    scores = compute_entropy(alphas, betas) - compute_conditional_entropy(alphas, betas)

    partitioned_indices = np.argpartition(scores, -batch_size)[-batch_size:]
    topk_indices = partitioned_indices[np.argsort(scores[partitioned_indices])[::-1]]
    candidate_scores = scores[topk_indices]

    return CandidateBatch(candidate_scores, topk_indices)

