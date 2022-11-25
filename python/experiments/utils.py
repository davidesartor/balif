from numpy.typing import NDArray
import numpy as np
from sklearn.metrics import precision_recall_curve


def extract_ap_matrix(sim_results) -> NDArray[np.float64]:
    """return matrix of average precision for each run and iteration"""
    ap_matrix = [
        [
            run_res["test"][f"after_{iter}_queries"]["ap"]
            for iter in range(run_res["n_iter"] + 1)
        ]
        for run_idx, run_res in sim_results.items()
    ]
    return np.array(ap_matrix)


def extract_scores_matrix(sim_results) -> NDArray[np.float64]:
    """return matrix of model scores for each run and iteration"""
    score_matrix = [
        [
            run_res["test"][f"after_{iter}_queries"]["scores"]
            for iter in range(run_res["n_iter"] + 1)
        ]
        for run_idx, run_res in sim_results.items()
    ]
    return np.array(score_matrix)


def extract_label_matrix(sim_results) -> NDArray[np.float64]:
    """return matrix of labels for each run and iteration"""
    label_matrix = [
        run_res["test"][f"labels"]
        for run_idx, run_res in sim_results.items()
    ]
    return np.array(label_matrix)


def extract_fbeta_matrix(sim_results, beta=1.0) -> NDArray[np.float64]:
    """return matrix of maximum f1 scores for each run and iteration"""
    def max_fbeta(labels, scores, beta=1.0):
        precision, recall, _ = precision_recall_curve(labels, scores)
        return max(np.divide((1+beta**2)*recall*precision, recall+beta**2*precision+1e-8))

    score_matrix = extract_scores_matrix(sim_results)
    label_matrix = extract_label_matrix(sim_results)

    f1_matrix = [[max_fbeta(run_labels, iteration_scores, beta) for iteration_scores in run_scores]
                 for run_labels, run_scores in zip(label_matrix, score_matrix)]
    return np.array(f1_matrix)

