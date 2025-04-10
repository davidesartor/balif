"""
This script implements BALD and BatchBALD for active learning 
using the closed formula for conditional and unconditional entropies
in the case of BALIF.
"""
import numpy as np
from scipy.special import digamma
from scipy.special import beta as BetaFunction
from typing import NamedTuple, List

class CandidateBatch(NamedTuple):
    scores: np.ndarray
    indices: np.ndarray

# ------------------- ClosedBALD -------------------#
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
    entropies = - means*np.log(means) - (1-means)*np.log(1-means)
    return entropies

def bald_scores(alphas, betas): 
    scores = compute_entropy(alphas, betas) - compute_conditional_entropy(alphas, betas)
    return scores
    
def get_bald_batch(alphas, betas, batch_size:int=1): 
    """
    Return top k samples with the highest BALD scores.
    
    Parameters
    ----------
    alphas : numpy.ndarray
        (n_samples, )
    betas : numpy.ndarray
        (n_samples, )
    batch_size : int, optional
        The default is 1.
    """
    scores = bald_scores(alphas, betas)

    partitioned_indices = np.argpartition(scores, -batch_size)[-batch_size:]
    topk_indices = partitioned_indices[np.argsort(scores[partitioned_indices])[::-1]]
    candidate_scores = scores[topk_indices]

    return CandidateBatch(candidate_scores, topk_indices)

def get_balif_bald_batch(alphas, betas, batch_size:int = 1): 
    """
    For each estimator, compute the BALD score for each sample. 
    Sum the scores over all trees. 
    Select the batch_size samples with the highest scores.

    Parameters
    ----------
    alphas : numpy.ndarray
        (n_samples, n_estimators)
    betas: numpy.ndarray
        (n_samples, n_estimators)
    batch_size : int, optional
        The default is 1.
    
    Returns 
    -------
    CandidateBatch
        A named tuple containing the scores and indices of the selected samples.
    """
    n_samples, n_trees = alphas.shape
    scores = np.zeros((n_samples, n_trees))
    for t in range(n_trees): 
        batch_t = bald_scores(alphas[:, t], betas[:, t]) # shape (n_samples, )
        scores[:, t] = batch_t
    scores = np.sum(scores, axis=1)                      # shape (n_samples, )


    
    topk_indices = np.argsort(scores)[-batch_size:][::-1] # shape (batch_size, )
    candidate_scores = scores[topk_indices]

    # these return different indices
    #    partitioned_indices = np.argpartition(scores, -batch_size)[-batch_size:]
    # topk_indices = partitioned_indices[np.argsort(scores[partitioned_indices])[::-1]]
    return CandidateBatch(candidate_scores, topk_indices)

# #------------------- ClosedBatchBALD -------------------#
def entropy_leaf(alphal:float, betal:float, samples:int):
    """
    Compute entropy of a leaf as H_l(y1,...,ylml) = - sum_{s=0}^{ml} comb(samples_in_leaf,s) p(s)log(p(s))
    where:
        - ml: samples in the leaf 
        - p(s) = B(alpha_l + s, beta_l + samples_in_leaf -s) / B(alpha_l, beta_l) 
               = exp(betaln(alpha_l + s, beta_l + samples_in_leaf -s) - betaln(alpha_l, beta_l))
    """
    s = np.arange(samples + 1)                                              # shape (ml+1, )
    binom_coeffs = comb(samples, s)                                         # shape (ml+1, )
    log_p_s = betaln(alphal+s, betal+samples-s) - betaln(alphal, betal)     # shape (ml+1, )
    p_s = np.exp(log_p_s)                                                   # shape (ml+1, )
    entropy = - np.sum(binom_coeffs*p_s*log_p_s)                            # shape (1,)
    return entropy

def joint_mutual_information(alphas, betas, leaves):
    """
    Compute the joint mutual information I(y1,...,yk; w) = H(y1,...,yk) - E_w[H(y1,...,yk|w)]
    where: 
        - H(y1,...,yk) = sum_l H_l(y1,...,ylml) where l are the leaves of the samples y1,...,yk
        - E_w[H(y1,...,yk|w)] = sum_k E_w[H(yk|wk)]

    Parameters
    ----------
    alphas : np.ndarray
        Alphahs parameters of w. Shape (k, n_estimators)
    betas : np.ndarray
        Betas parameters of w. Shape (k, n_estimators)
    leaves : np.ndarray
        Leaves idxs for each sample. Shape (k, n_estimators)

    Returns
    -------
    float
        Mutual information I(y1,...,yk; w) for each estimator. Shape (n_estimators,)
    """
    n_estimators = alphas.shape[1]
    # joint conditional entropy 
    samplesizes = alphas + betas         # (k, n_estimators)
    means = alphas / samplesizes         # (k, n_estimators)
    cond_entropies = - means*digamma(alphas+1) - (1-means)*digamma(betas+1) + digamma(samplesizes+1) 
    joint_cond_entropies = np.sum(cond_entropies, axis=0)
    #NOTE: to sum across all axis if you want to sum inside here over estimators 

    # joint entropy (non so come non iterare sugli alberi)
    joint_entropies = np.zeros(n_estimators)
    for t in range(n_estimators): 
        leavest = leaves[:, t]
        unique_leaves, counts = np.unique(leavest, return_counts=True)

        # extract one representative alpha and beta for each leaf 
        unique_alphas = np.empty_like(unique_leaves, dtype=float)
        unique_betas = np.empty_like(unique_leaves, dtype=float)
        for i, leaf in enumerate(unique_leaves): 
            idx = np.where(leavest == leaf)[0][0]   # get first index of the leaf
            unique_alphas[i] = alphas[idx, t]      # shape (ml,)
            unique_betas[i] = betas[idx, t]        # shape (ml,)
        
        # compute entropy for the tree as sum over the leaves
        jentropy_tree = 0.0
        for alpha_l, beta_l, samples_in_leaf in zip(unique_alphas, unique_betas, counts): 
            jentropy_tree += entropy_leaf(alpha_l, beta_l, samples_in_leaf)
        
        joint_entropies[t] = jentropy_tree

    #NOTE: if you want to sum across all trees you can do it here

    assert joint_entropies.shape == (n_estimators,)
    assert joint_cond_entropies.shape == (n_estimators,)
    joint_mi = joint_entropies - joint_cond_entropies
    assert np.all(joint_mi >= 0), f"Joint MI is negative: {joint_mi}"
    return joint_mi

def batchbald_batch(alphas, betas, leaves, k:int=1): 
    """
    Greedily selects the k most informative samples for batchBALD.
    """
    n_samples, n_estimators = alphas.shape
    A = np.empty(k, dtype=int)  
    A_scores = np.empty(k, dtype=float)     

    selected = np.zeros(n_samples, dtype=bool) # keep track of selected samples 
    
    for i in range(k):
        scores = -1 * np.ones((n_samples, n_estimators)) 
        alphas_sel = alphas[selected]                   # (i, n_estimators)
        betas_sel = betas[selected]                     # (i, n_estimators)
        leaves_sel = leaves[selected]                   # (i, n_estimators)

        candidates = np.setdiff1d(np.arange(n_samples), A[:i], assume_unique=True)
        for j in candidates: 
            alphas_batch = np.concatenate((alphas_sel, alphas[j:j+1]))  # (i+1, n_estimators)
            betas_batch = np.concatenate((betas_sel, betas[j:j+1]))
            leaves_batch = np.concatenate((leaves_sel, leaves[j:j+1]))

            mi = joint_mutual_information(alphas_batch, betas_batch, leaves_batch)
            scores[j] = mi

            
        # sum over trees 
        scores = np.sum(scores, axis=1)
        
        scores[selected] = -np.inf
        best_candidate = np.argmax(scores)
        best_score = scores[best_candidate]
        A[i] = best_candidate
        A_scores[i] = best_score

        # update selected samples
        selected[best_candidate] = True
    
    return A


if __name__ == '__main__':
    n_samples, n_estimators = 100, 3

    # Generate leaf indices
    leaf_idxs = np.random.randint(0, 10, size=(n_samples, n_estimators))

    # Generate shared (alpha, beta) for each unique leaf
    unique_leaf_ids = np.unique(leaf_idxs)
    leaf_to_params = {
        leaf: (np.random.randint(1, 10), np.random.randint(1, 10))
        for leaf in unique_leaf_ids
    }

    # Build alpha and beta arrays with shared values for same leaf
    alphas = np.zeros_like(leaf_idxs)
    betas = np.zeros_like(leaf_idxs)
    for i in range(n_samples):
        for j in range(n_estimators):
            leaf = leaf_idxs[i, j]
            alpha, beta = leaf_to_params[leaf]
            alphas[i, j] = alpha
            betas[i, j] = beta

    batch = batchbald_batch(alphas, betas, leaf_idxs, k=10)
    print("selected", batch)

        


    
