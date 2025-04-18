from joblib import Parallel, delayed
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

import odds_datasets
import balif


def run_sim(
    dataset,
    batch_size,
    strategy,
    seed,
):
    # load dataset
    X, y = odds_datasets.load(dataset)
    contamination = y.mean()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y
    )

    # duplicate trainset (copies = batch_size)
    X_train = np.concatenate([X_train] * batch_size, axis=0)
    y_train = np.concatenate([y_train] * batch_size, axis=0)

    # fit the unsupervised model
    model = balif.BADIForest(
        contamination=contamination,
        random_state=seed,
    ).fit(X_train)

    # get and save unsupervised average precision
    scores_train = model.decision_function(X_train)
    scores_test = model.decision_function(X_test)
    avp_train = [average_precision_score(y_train, scores_train)]
    avp_test = [average_precision_score(y_test, scores_test)]

    # run the simulation
    iterations = int(np.floor(len(X_train) / batch_size))
    queriable = np.ones(len(X_train), dtype=bool)
    for _ in tqdm(range(iterations), f"{dataset}: {strategy} bs={batch_size}"):
        # get the queries indices
        if strategy.startswith("independent"):
            _, interest_method = strategy.split("_")
            idxs = balif.active_learning.get_queries_independent(
                model=model,
                X=X_train,
                interest_method=interest_method,  # type: ignore
                batch_size=batch_size,
                mask=queriable,
            )
        elif strategy.startswith("worstcase"):
            _, interest_method = strategy.split("_")
            idxs = balif.active_learning.get_queries_worstcase(
                model=model,
                X=X_train,
                interest_method=interest_method,  # type: ignore
                batch_size=batch_size,
                mask=queriable,
            )
        elif strategy == "batchbald":
            idxs = balif.active_learning.get_queries_batchbald(
                model=model,
                X=X_train,
                batch_size=batch_size,
                mask=queriable,
            )
        elif strategy == "random":
            idxs = np.random.choice(
                a=np.arange(len(X_train))[queriable],
                size=batch_size,
                replace=False,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # update the model with the queried samples
        model.update(X_train[idxs, :], y_train[idxs])
        queriable[idxs] = False

        # get the average precision scores
        scores_train = model.decision_function(X_train)
        scores_test = model.decision_function(X_test)
        avp_train.append(average_precision_score(y_train, scores_train))
        avp_test.append(average_precision_score(y_test, scores_test))

    if np.any(queriable):
        # query the remaining samples
        idxs = np.arange(len(X_train))[queriable]
        model.update(X_train[idxs, :], y_train[idxs])
        queriable[idxs] = False

        # get the average precision scores
        scores_train = model.decision_function(X_train)
        scores_test = model.decision_function(X_test)
        avp_train.append(average_precision_score(y_train, scores_train))
        avp_test.append(average_precision_score(y_test, scores_test))

    # save results
    assert not queriable.any(), "Some points have not been queried"
    file_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(file_dir, "results", dataset, f"batch_size_{batch_size}", strategy)
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(
        f"{save_dir}/seed_{seed}",
        avp_train=np.array(avp_train),
        avp_test=np.array(avp_test),
    )


if __name__ == "__main__":
    # sweep config parameters
    seeds = list(range(10))
    batch_sizes = [1, 5, 10]
    strategies = (
        ["random"]
        + ["independent_bald", "worstcase_bald"]
        + ["independent_margin", "worstcase_margin"]
        + ["independent_anom", "worstcase_anom"]
        + ["batchbald"]
    )

    # run the simulations
    Parallel(n_jobs=48)(
        delayed(run_sim)(
            dataset=dataset,
            batch_size=batch_size,
            strategy=strategy,
            seed=seed,
        )
        for dataset in odds_datasets.datasets_names
        for strategy in strategies
        for batch_size in batch_sizes
        for seed in seeds
    )

    # collect multiple seeds in a single file
    for i, dataset in enumerate(odds_datasets.datasets_names):
        for strategy in strategies:
            for batch_size in batch_sizes:
                file_dir = os.path.dirname(os.path.abspath(__file__))
                save_dir = os.path.join(
                    file_dir, "results", dataset, f"batch_size_{batch_size}", strategy
                )
                files = [f"{save_dir}/{f}" for f in os.listdir(save_dir)]
                avp_test = np.stack([np.load(f)["avp_test"] for f in files], axis=-1)
                avp_train = np.stack([np.load(f)["avp_train"] for f in files], axis=-1)
                np.savez_compressed(save_dir, avp_test=avp_test, avp_train=avp_train)
                for f in files:
                    os.remove(f)
                os.rmdir(save_dir)