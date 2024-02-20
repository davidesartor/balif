from __future__ import annotations
from typing import NamedTuple, Optional
from numpy.typing import NDArray
import numpy as np

from dataclasses import dataclass, field

from scipy.stats import beta
from .utils.binarytree import expected_isolation_depth
from .isolationforest import IsolationForest, IsolationTree


class BetaDistr(NamedTuple):
    alpha: NDArray[np.float64]
    beta: NDArray[np.float64]

    @classmethod
    def from_mean_and_samplesize(
        cls, mean: NDArray[np.float64], samplesize: NDArray[np.float64]
    ) -> BetaDistr:
        return cls(mean * samplesize, (1 - mean) * samplesize)

    @property
    def mean(self) -> NDArray[np.float64]:
        return self.alpha / (self.alpha + self.beta)

    @property
    def samplesize(self) -> NDArray[np.float64]:
        return self.alpha + self.beta

    def pdf(self, x: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return beta.pdf(x, self.alpha, self.beta)

    @staticmethod
    def combine(distrs: list[BetaDistr], mode="naive") -> BetaDistr:
        """Combine a list of beta distributions."""
        if mode == "naive":
            return BetaDistr.from_mean_and_samplesize(
                mean=np.mean([distr.mean for distr in distrs], axis=0),
                samplesize=np.sum([distr.samplesize for distr in distrs], axis=0),
            )
        else:
            raise ValueError(f"Unknown combination mode: {mode}")


@dataclass(eq=False)
class BalifTree(IsolationTree):
    """Modified version of the isolation tree for active learning."""

    # additional node properties
    alphas: NDArray[np.float64] = field(init=False, repr=False)
    betas: NDArray[np.float64] = field(init=False, repr=False)

    def fit(self, data: NDArray[np.float64], seed: Optional[int] = None) -> None:
        """Create the isolation tree from the data."""
        super().fit(data, seed=seed)
        offset = {"haldane": 1e-8, "balanced": 0.1, "jeffreys": 0.5, "bayes": 1.0}["balanced"]
        base_pred = 2 ** (-self.corrected_depths / self.c_norm)
        size = offset / np.minimum(base_pred, 1 - base_pred)
        self.alphas, self.betas = base_pred * size, (1 - base_pred) * size

    def predict_distr(self, data: NDArray[np.float64]) -> BetaDistr:
        leafs = self.isolation_nodes_idxs(data)
        return BetaDistr(self.alphas[leafs], self.betas[leafs])

    def predict(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.predict_distr(data).mean

    def update(
        self,
        *args,
        inlier_data: Optional[NDArray[np.float64]],
        outlier_data: Optional[NDArray[np.float64]],
    ) -> None:
        """Update the model with new labelled data points."""
        if inlier_data is not None:
            if inlier_data.ndim == 1:
                inlier_data = inlier_data[None, :]
            terminal_nodes = self.isolation_nodes_idxs(inlier_data)
            self.betas[terminal_nodes] += 1.0

        if outlier_data is not None:
            if outlier_data.ndim == 1:
                outlier_data = outlier_data[None, :]
            terminal_nodes = self.isolation_nodes_idxs(outlier_data)
            self.alphas[terminal_nodes] += 1.0


@dataclass(eq=False)
class Balif(IsolationForest):
    """Baeysian Active learning isolation forest."""

    trees: list[BalifTree] = field(init=False, repr=False, default_factory=list)

    def create_tree(self, data: NDArray[np.float64]) -> BalifTree:
        """Create a single estimator."""
        samplesize = min((self.max_bagging_samples, data.shape[0]))
        subsample = data[np.random.randint(data.shape[0], size=samplesize)]
        return BalifTree(data=subsample, hyperplane_components=self.hyperplane_components)

    def update(
        self,
        *,
        inlier_data: Optional[NDArray[np.float64]] = None,
        outlier_data: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Update the model with new labelled data points."""
        for tree in self.trees:
            tree.update(inlier_data=inlier_data, outlier_data=outlier_data)

    def predict_distr(self, data: NDArray[np.float64]) -> BetaDistr:
        """Predict the anomaly distr for each sample in the data."""
        return BetaDistr.combine([tree.predict_distr(data) for tree in self.trees])

    def predict(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict the anomaly score for each sample in the data."""
        return self.predict_distr(data).mean

    def interest(self, data: NDArray[np.float64], strategy: str = "score"):
        """Compute the interest of the data points."""
        if strategy == "score":
            return self.predict(data)
        elif strategy == "random":
            return np.random.rand(data.shape[0])
        elif strategy == "margin":
            return self.predict_distr(data).pdf(0.5)
        else:
            raise ValueError(f"Unknown interest strategy: {strategy}")
