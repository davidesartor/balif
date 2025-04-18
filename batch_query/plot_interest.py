import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os 
import balif


if __name__ == "__main__":
    a = 1 + 10 ** np.linspace(-1, 2, 100)
    b = 1 + 10 ** np.linspace(-1, 2, 100)
    A, B = np.meshgrid(a, b)
    a0, b0 = 15, 10
    delta_nu = [30, 60, 90]

    plt.figure(figsize=(15, 4))
    for i, method in enumerate(["bald", "margin", "anom"]):
        plt.subplot(1, 3, i + 1)
        beliefs = balif.BetaDistr(a=A, b=B)
        scores = balif.interest(beliefs, method) # type: ignore
        plt.contourf(A, B, scores, levels=20, cmap="viridis")
        plt.colorbar()

        plt.plot(a0, b0, "r*", markersize=10)
        for d in delta_nu:
            mu = np.linspace(0, 1, 100)
            ss = a0 + b0 + d
            a, b = ss * mu, ss * (1 - mu)
            plt.plot(a, b, "r:")

            a = a0 + np.arange(d + 1)
            b = b0 + np.arange(d + 1)[::-1]
            plt.plot(a, b, "r-", linewidth=3)

        plt.title(method)
        plt.xlabel("$\\alpha$")
        plt.ylabel("$\\beta$")
        # plt.xscale("log")
        # plt.yscale("log")
        plt.xlim(1, A.max())
        plt.ylim(1, B.max())
        plt.gca().set_aspect("equal", adjustable="box")

        # inner plot
        axins = inset_axes(plt.gca(), width="50%", height="20%", loc="upper right")
        for d in delta_nu:
            mu = np.linspace(0, 1, 100)
            ss = a0 + b0 + d
            a, b = ss * mu, ss * (1 - mu)
            beliefs = balif.BetaDistr(a=a, b=b)
            scores = balif.interest(beliefs, method) # type: ignore
            plt.plot(mu, scores, "r:")

            a = a0 + np.arange(d + 1)
            b = b0 + np.arange(d + 1)[::-1]
            mu = a / (a + b)
            beliefs = balif.BetaDistr(a=a, b=b)
            scores = balif.interest(beliefs, method) # type: ignore
            axins.plot(mu, scores, "r-", linewidth=3)
            break

        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_xlabel("$\\mu$")

    plt.tight_layout()
    file_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(file_dir, "figures")
    save_path = os.path.join(save_dir, f"bald_margin_anom.pdf")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")