import os
import numpy as np
import matplotlib.pyplot as plt
from balif import odds_datasets

if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.abspath(__file__))

    # compare worstcase and batchbald
    for batch_size in [10, 5, 1]:
        plt.figure(figsize=(8, 12))
        for i, dataset in enumerate(
            odds_datasets.small_datasets_names
            + odds_datasets.medium_datasets_names
            + odds_datasets.large_datasets_names
            + ["legend"]
        ):
            plt.subplot(6, 3, i + 1)
            for strategy, label, color, linestyle in [
                ("worstcase_bald", "bald worstcase", "tab:blue", "-"),
                ("independent_bald", "bald independent", "tab:blue", ":"),
                ("worstcase_margin", "margin worstcase", "tab:orange", "-"),
                ("independent_margin", "margin independent", "tab:orange", ":"),
                ("batchbald", "batchbald", "tab:green", "-"),
                ("random", "random", "black", ":"),
            ]:
                if dataset == "legend":
                    plt.plot([], [], label=label, color=color, linestyle=linestyle)
                else:
                    # load simulation data
                    sim_file = os.path.join(
                        file_dir, "results", dataset, f"batch_size_{batch_size}", strategy
                    )
                    sim = np.load(f"{sim_file}.npz")["avp_test"]

                    x = np.linspace(0, 100, len(sim), endpoint=True)
                    value = sim.mean(-1)
                    lb = value - sim.std(-1) * 1.96 / np.sqrt(sim.shape[-1])
                    ub = value + sim.std(-1) * 1.96 / np.sqrt(sim.shape[-1])

                    # plot simulation data
                    plt.plot(x + 1, value, label=label, color=color, linestyle=linestyle)
                    plt.fill_between(x + 1, lb.clip(0, 1), ub.clip(0, 1), alpha=0.1, color=color)

                if dataset == "legend":
                    plt.legend(loc="center", fontsize=12, frameon=False)
                    plt.axis("off")
                else:
                    plt.title(f"{dataset}")
                    plt.grid(True)
                    plt.xscale("log")
                    plt.xticks([1, 2, 6, 11, 101], ["0%", "1%", "5%", "10%", "100%"])
                    plt.xlim(1, 11)
                    # plt.ylim(0, 1)
                    if i % 3 == 0:
                        plt.ylabel("AP")
                    if i >= 15:
                        plt.xlabel("Labelling Budget")

        plt.tight_layout()
        save_dir = os.path.join(file_dir, "figures")
        save_path = os.path.join(save_dir, f"ablation_variants_{batch_size}.pdf")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
