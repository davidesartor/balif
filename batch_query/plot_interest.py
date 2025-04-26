import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os 
from balif import active_learning, BetaDistr


if __name__ == "__main__":
    a = 1 + 10 ** np.linspace(-1, 2, 1000)
    b = 1 + 10 ** np.linspace(-1, 2, 1000)
    A, B = np.meshgrid(a, b)
    a0, b0 = 15, 10
    delta_nu = [30, 60, 90]

    plt.figure(figsize=(9, 2.5))
    for i, method in enumerate(["bald", "margin", "anom"]):
        plt.subplot(1, 3, i + 1)
        beliefs = BetaDistr(a=A, b=B)
        scores = active_learning.interest(beliefs, method) # type: ignore
        if method == "bald":
            scores = np.log10(scores).clip(-2.5, -1.20)
        plt.contourf(A, B, scores, levels=20, cmap="coolwarm")
        if method == "bald":
            ticks = [-1.20, -1.36, -1.52, -1.68, -1.84, -2.00, -2.16, -2.32, -2.48]
            cbar = plt.colorbar(ticks=ticks)
            cbar.ax.set_yticklabels([f"{10*10**t:.2f}" for t in ticks])
        else:
            plt.colorbar()

        plt.plot(a0, b0, "k*", markersize=6)
        for d in delta_nu:
            mu = np.linspace(0, 1, 100)
            ss = a0 + b0 + d
            a, b = ss * mu, ss * (1 - mu)
            plt.plot(a, b, "k:")

            a = a0 + np.arange(d + 1)
            b = b0 + np.arange(d + 1)[::-1]
            plt.plot(a, b, "k-", linewidth=2)

        plt.title(method.upper())
        plt.xlabel("$\\alpha$")
        if i == 0:
            plt.ylabel("$\\beta$")
        # plt.xscale("log")
        # plt.yscale("log")
        plt.xlim(1, A.max())
        plt.ylim(1, B.max())
        plt.xticks([25, 50, 75, 100])
        plt.yticks([25, 50, 75, 100])
        plt.gca().set_aspect("equal", adjustable="box")

        # inner plot
        axins = inset_axes(plt.gca(), width="50%", height="20%", loc="upper right")
        for d in delta_nu:
            mu = np.linspace(0, 1, 100)
            ss = a0 + b0 + d
            a, b = ss * mu, ss * (1 - mu)
            beliefs = BetaDistr(a=a, b=b)
            scores = active_learning.interest(beliefs, method) # type: ignore
            plt.plot(mu, scores, "k:")

            a = a0 + np.arange(d + 1)
            b = b0 + np.arange(d + 1)[::-1]
            mu = a / (a + b)
            beliefs = BetaDistr(a=a, b=b)
            scores = active_learning.interest(beliefs, method) # type: ignore
            axins.plot(mu, scores, "k-", linewidth=2)
            break

        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_xlabel("$\\mu$")
        if method == "bald":
            axins.set_ylim(scores.min()*0.9, scores.max()*1.1)
    
    plt.tight_layout()
    file_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(file_dir, "figures")
    save_path = os.path.join(save_dir, f"bald_margin_anom.pdf")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")