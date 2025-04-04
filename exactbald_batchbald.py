"""
This script implements BALD and BatchBALD for active learning 
using the exact formula for conditional and unconditional entropies
in the case of BALIF.
"""
import numpy as np
from scipy.special import digamma
from scipy.special import beta as BetaFunction
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
    entropies = - means*np.log(means) - (1-means)*np.log(1-means)
    return entropies

def bald_scores(alphas, betas): 
    scores = compute_entropy(alphas, betas) - compute_conditional_entropy(alphas, betas)
    return scores
    

def get_exactbald_batch(alphas, betas, batch_size:int=1): 
    scores = bald_scores(alphas, betas)

    partitioned_indices = np.argpartition(scores, -batch_size)[-batch_size:]
    topk_indices = partitioned_indices[np.argsort(scores[partitioned_indices])[::-1]]
    candidate_scores = scores[topk_indices]

    return CandidateBatch(candidate_scores, topk_indices)


#------------------- ExactBatchBALD -------------------#

def compute_leaf_entropy(alpha_l:float, beta_l:float, samples_in_leaf:int): 
    """
    Compute H_l(y_l_1, ..., y_l_ml) = - - sum_s=0*^ml p(s)*log2(p(s))

    where p(s) = p(s-1) * (alpha_l + s-1)*(beta_l - 1 + ml -s) / ((s+1) * (ml-s))
    and p(0) = B(alpha_l, beta_l + ml)/B(alpha_l, beta_l)

    Parameters
    ----------
    alpha_l : float
        alpha parameter of the anomaly rate of leaf l
    beta_l : float
        beta parameter of the anomaly rate of leaf l
    
    # change in the log form
    # vectorial way 
    # code the Beta function directly? 
    """
    p = BetaFunction(alpha_l, beta_l + samples_in_leaf) / BetaFunction(alpha_l, beta_l)
    H_l = -p * np.log(p)
    
    for s in range(1, samples_in_leaf+1): 
        fs = (alpha_l+s)*(beta_l -1+samples_in_leaf-s)/((s+1)*(samples_in_leaf-s))
        p = p * fs 
        H_l += -p * np.log(p)
    
    return H_l

def compute_joint_entropy(alphas_leaves, betas_leaves, samples_in_leaf): 
    """
    Compute H(y1, ..., yp) = sum_l H_l(y_l_1, ..., y_l_ml) where m_l is the number of samples in leaf l.

    Parameters
    ----------
    alphas_leaves : numpy.ndarray -> alpha di ogni foglia in cui finiscono i punti
        (n_leaves, )
    betas_leaves : numpy.ndarray
        (n_leaves, )
    samples_in_leaf : numpy.ndarray
        (n_leaves, )
    """
    n_leaves = alphas_leaves.shape[0]
    joint_entropy = 0.0
    for l in range(n_leaves):
        alpha_l = alphas_leaves[l]
        beta_l = betas_leaves[l]
        samples_in_leaf_l = samples_in_leaf[l]
        joint_entropy += compute_leaf_entropy(alpha_l, beta_l, samples_in_leaf_l)
    return joint_entropy


def compute_conditional_joint_entropy(alphas, betas): 
    """
    E_w[H(y1, ..., yk|w)] = sum_i E_w[H(yi|wi)] assuming y1|w1, ..., yp|wk independent

    Parameters
    ----------
    alphas : numpy.ndarray
        (k, )
    betas : numpy.ndarray
        (k, )   
    """
    single_conditional_entropies = compute_conditional_entropy(alphas, betas)       # shape (k, )
    joint_conditional_entropy = np.sum(single_conditional_entropies, axis=0)        # float      
    return joint_conditional_entropy


def batchbald_scores(alphas, betas, group_by_leaf):
    """
    Given parameters w and set of p points 1,...,k, compute I(y1, ..., yk;w) 

    Parameters
    ----------
    alphas : numpy.ndarray
        (k, )
    betas : numpy.ndarray
        (k, )
    group_by_leaf : numpy.ndarray. The group_by_leaf[i] is the leaf index of the i-th point.
        (k, )

    # Returns
    # -------
    # numpy.ndarray
    #     (p, )
    """
    # group points by leaf 
    unique_leaves, counts = np.unique(group_by_leaf, return_counts=True)

    # for each leaf, assuming that each point in the leaf has the same alpha and beta, 
    # thake the first alpha and beta 
    alphas_leaves = np.array([alphas[group_by_leaf == leaf][0] for leaf in unique_leaves])
    betas_leaves = np.array([betas[group_by_leaf == leaf][0] for leaf in unique_leaves])

    # compute the joint entropy using leaves
    joint_entropy_per_leaf = compute_joint_entropy(alphas_leaves, betas_leaves, counts) # shape (n_leaves, )

    # sum over leaves 
    joint_entropy = np.sum(joint_entropy_per_leaf) 

    # expand the joint entropies to each point 
    indices = np.searchsorted(unique_leaves, group_by_leaf)
    scores = joint_entropy_per_leaf[indices]

    scores = - compute_conditional_joint_entropy(alphas, betas) # shape ()
    return scores

def get_batchbald_batch(alphas, betas, group_by_leaves, k:int = 1): 
    """
    Greedily select k points that maximize the batchbald score.

    Initialize A = {} for storing selected points. 
    For each iteration i from 1 to k: 
        1. For each tree:
            For each point j not in A:  
                a. Compute I_t(y1, ..., yk-1, yj; w)
        2. Sum the scores over all tree to get I(y1, ..., yk-1, yj; w)
        3. Select the point with the highest score and add it to A.

    Parameters
    ----------  
    alphas : numpy.ndarray
        (n_samples, n_trees)
    betas : numpy.ndarray
        (n_samples, n_trees)
    group_by_leaves : numpy.ndarray
        The group_by_leaves[i] is the leaf index of the i-th point.
        (n_samples, n_trees)
    k : int
        Number of samples to select.
    """    
    A = []  
    # A_scores = np.zeros(k)

    n_trees = alphas.shape[1]

    # iterate over k points
    for i in range(k): 

        scores_over_tree = np.zeros((i+1, alphas.shape[0], n_trees))
        
        # for each tree, for each point compute the scores given the previous points in A
        for tree in range(n_trees): 

            # for each point not in A, retrieve alphas and betas of previous points and candidate point
            # and compute the scores 
            # gather believes should return a (n_samples, )
            alphas_t, betas_t = alphas[:, tree], betas[:, tree]
            group_by_leaves_t = group_by_leaves[:, tree]

            # pass the model and use gather remove loop over trees--------------------------------------------------------------------
            alphas_in_A = alphas_t[A]
            betas_in_A = betas_t[A]
            group_by_leaves_in_A = group_by_leaves_t[A]

            # scores_t = np.zeros((i+1, alphas.shape[0]-len(A)))

            # move the point to the beginning and take the ordering maps
            # so we avoid to chekc if j in A or not 
            for j in range(alphas.shape[0]): 
                if j not in A: 
                    # concatenate alphas_in_A and alphas_t[j]
                    alphas_candidate_batch = np.concatenate((alphas_in_A, alphas_t[j].reshape(1, -1)), axis=0) # shape (i+1, )
                    betas_candidate_batch = np.concatenate((betas_in_A, betas_t[j].reshape(1, -1)), axis=0) # shape (i+1, )
                    group_by_leaves_candidate_batch = np.concatenate((group_by_leaves_in_A, group_by_leaves_t[j].reshape(1, -1)), axis=0)
                    assert alphas_candidate_batch.shape[0] == i+1

                    mutual_info = batchbald_scores(alphas_candidate_batch, betas_candidate_batch, group_by_leaves_candidate_batch)      # shape (i+1, ) 
                    scores_over_tree[:, j, tree] = mutual_info
                else: 
                    # if the candidate is already selected, we don't want it to be selected again
                    # so we set the score to -inf
                    scores_over_tree[:, j, tree] = -np.inf 
        

        # sum over trees
        scores_over_tree = np.sum(scores_over_tree, axis=2)         # shape (i+1, n_samples)

        # take the argmax (since points in A have scores -inf, they will be ignored)
        best_candidate = np.argmax(scores_over_tree[i, :])
        best_score = scores_over_tree[i, best_candidate]



            
            

        


    
