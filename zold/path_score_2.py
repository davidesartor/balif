from __future__ import annotations

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
    def p_ki_matrix(k0: int) -> NDArray[np.float64]:
        """Compute the probability of having a node size of ki at layer i as a 2d Array.

        Returns
        -------
        p(i, ki): 2DArray[np.float64]
            2d matrix containing the probability of having a node size of ki at layer i.
        """
        P_trans = PathScorer.transition_matrix(k0)
        probabilities = np.zeros(shape=(int(np.log2(k0) + 1), k0 + 1), dtype=np.float64)
        probabilities[0, k0] = 1.0
        for level in range(1, int(np.log2(k0) + 1)):
            probabilities[level] = np.matmul(P_trans, probabilities[level - 1])  # type: ignore
        return probabilities

    def probability_of_ki(
        self, i: int | tuple[int, int], ki: int | tuple[int, int]
    ) -> np.float64:
        """Compute the probability of having a node size of ki at layer i.

        Parameters
        ----------
        i : int or tuple[int,int]
            layer index or range of layer indices (extremes included)
        ki : int or tuple[int,int]
            node size at layer i or range of node sizes at layer i (extremes included)

        Returns
        -------
        p(i, ki): np.float64 or 1DArray[np.float64] or 2DArray[np.float64]

        Raises
        ------
        ValueError
            if either i or ki is a tuple of length different from 2.
        """
        if isinstance(i, int):
            i = (i, i)
        if isinstance(ki, int):
            ki = (ki, ki)
        if not (len(i) == len(ki) == 2):
            raise ValueError("i and ki must be tuples of length 2.")
        return PathScorer.p_ki_matrix(self.k0)[i[0] : i[1]+1, ki[0] : ki[1]+1]

    def prob_transition(
        self, ki: int | tuple[int, int], kiplus1: int | tuple[int, int]
    ) -> np.float64:
        """Compute the probability of a transition from a node size of ki to a node size of ki+1.

        Parameters
        ----------
        ki : int or tuple[int,int]
            node size at layer i or range of node sizes at layer i (extremes included)

        kiplus1 : int or tuple[int,int]
            node size at layer i+1 or range of node sizes at layer i+1 (extremes included)

        Returns
        -------
        P(ki+1, ki): np.float64 or 1DArray[np.float64] or 2DArray[np.float64]

        Raises
        ------
        ValueError
            if either ki or kiplus1 is a tuple of length different from 2.
        """
        if isinstance(ki, int):
            ki = (ki, ki)
        if isinstance(kiplus1, int):
            kiplus1 = (kiplus1, kiplus1)
        if not (len(ki) == len(kiplus1) == 2):
            raise ValueError("ki and kiplus1 must be tuples of length 2.")
        return PathScorer.transition_matrix(self.k0)[
            ki[0] : ki[1]+1, kiplus1[0] : kiplus1[1]+1
        ]

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
        res = np.sum(self.probability_of_ki(i=len(kis) - 1, ki=(1,kis[-1]-1)))
        tail = [kis[-1]]
        prob_tail = np.float64(1.0)
        for i, ki in reversed(list(enumerate(kis[:-1]))):
            # sum for all k in (1,ki-1) the p(ki = k) * P(k->ki+1) * P(ki+1,ki+2) * ... * P(kn-1,kn)
            res += np.sum(
                self.probability_of_ki(i=len(kis) - 1, ki=(1,ki-1))
                * self.prob_transition(ki=(1,ki-1), kiplus1=tail[0])
                * prob_tail
            )
            tail = [ki] + tail
            prob_tail *= self.prob_transition(ki, tail[0])
        return float(res)
