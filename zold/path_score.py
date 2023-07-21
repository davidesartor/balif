from dataclasses import dataclass
from functools import cache
from typing import Sequence
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PathScorer:
    """Compute the score of a path in an isolation tree.

    Parameters
    ----------
    k0 : int
        size of the sample used to construct the isolation tree.
    """

    k0: int

    @staticmethod
    @cache
    def transition_matrix(k0: int) -> NDArray[np.float64]:
        """Create the transition matrix for the Markov chain of node sizes.

        P(j,i) is the probability of going to a node size of i from a node size of j.
        If i>1, then P(j,i) = 1/(i-1) for j<i. P(j,i) = 0 otherwise.
        If i=1, then P(j,i) = 1 for j=1. P(j,i) = 0 otherwise.

        Parameters
        ----------
        k0 : int
            size of the sample used to construct the isolation tree, and maximum node size of the transition matrix.

        Returns
        -------
        P: 2DArray[np.float64]
            2d matrix with entries P[j,i] for i,j in [0,k0] containing the probability of going from a node size of i to a node size of j.
        """
        trans_matrix = np.zeros(shape=(k0 + 1, k0 + 1), dtype=np.float64)
        for i in range(2, k0 + 1):
            trans_matrix[1:i, i] = 1 / (i - 1)
        trans_matrix[0, 0] = 1
        trans_matrix[1, 1] = 1
        return trans_matrix

    @staticmethod
    @cache
    def probability_of_ki(k0: int) -> NDArray[np.float64]:
        """Compute the probability of having a node size of ki at layer i.

        Returns
        -------
        p(i, ki): 2DArray[np.float64]
            2d matrix containing the probability of having a node size of ki at layer i.
        """
        P_trans = PathScorer.transition_matrix(k0)
        probabilities = np.zeros(shape=(int(np.log2(k0)+1), k0 + 1), dtype=np.float64)
        probabilities[0, k0] = 1.0
        for level in range(1, int(np.log2(k0)+1)):
            probabilities[level] = np.matmul(P_trans, probabilities[level - 1])  # type: ignore
        return probabilities
    
    def prob_tail(self, ki_to_kn: Sequence[int]) -> np.float64:
        """Compute the probability of a tail of node sizes, meaning P(ki, ki+1, ..., kn | ki)

        Parameters
        ----------
        ki_to_kn : Sequence[int]
            list of node sizes, ordered from the size of layer i to the size of layer n.

        Returns
        -------
        np.float64
            probability of having such a tail.
        """
        if len(ki_to_kn) <= 1:
            return np.float64(1.0)
        
        P_trans = PathScorer.transition_matrix(self.k0)
        factor = np.float64(1.0)
        for ki, kiplus1 in zip(ki_to_kn[:-1], ki_to_kn[1:]):
            factor *= P_trans[kiplus1, ki]
        return factor

    def score(self, kis: Sequence[int]) -> float:
        """Return the anomaly score of a path as the probability of having a more imbalanced path.
        The imbalance of a path is defined by ordering sequences of node sizes (considered from leaf to root).
        i.e: for a tree of depth 4, the path k3,k2,k1,k0 = [1,1,3,4] is more imbalanced than [1,2,2,4].
        Following this ordering preserves the ordering of anomaly scores as computed by the isolation forest.
        But allows distinctions between paths that reach node sizes of 1 (and stop) at same depth.

        Parameters
        ----------
        kis : Sequence[int]
            sequence of node sizes, ordered from the size of the root layer to the size of the leaf layer.

        Returns
        -------
        score: float
            the score of the path as the probability of having a more imbalanced path.
        """
        P_i_ki = PathScorer.probability_of_ki(self.k0)
        P_trans = PathScorer.transition_matrix(self.k0)
        res = np.float64(0.0)
        tail = []
        for ki in kis[::-1]:
            prob_tail = self.prob_tail(tail)
            factor = P_trans[tail[0], 1:ki] * prob_tail if len(tail)>0 else np.ones(shape=(ki-1,))
            res += np.dot(P_i_ki[-len(tail)-1, 1:ki],factor)
            tail = [ki] + tail
        return float(res)
