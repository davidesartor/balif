import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import argparse
from tqdm.auto import tqdm
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import comet_ml
from sklearn.metrics import average_precision_score as avg_precision

import odds_datasets
from balif import Balif


@eqx.filter_jit
def run_fn(model_config, data, labels, key):
    def scan_body(carry, key):
        key_score, key_query, key_update = jr.split(key, 3)
        model, queriable = carry

        scores = model.score(data, key=key_score)

        interests = model.interest(data, key=key_query)
        query_idx = jnp.where(queriable, interests, interests.min()).argmax()
        queriable = queriable.at[query_idx].set(False)
        point, is_anomaly = data[query_idx], labels[query_idx]

        model = model.register(point, is_anomaly=is_anomaly, key=key_update)
        return (model, queriable), scores

    samples, dim = data.shape
    iterations = 1 + samples // 10
    rng_fit, rng_steps = jr.split(key)
    model = Balif(**model_config)
    model = model.fit(data, key=rng_fit)

    queriable = jnp.ones(samples, dtype=bool)
    _, scores = jax.lax.scan(
        scan_body, (model, queriable), jr.split(rng_steps, iterations)
    )
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--bootstrap", type=bool, default=True)
    parser.add_argument("--standardize", type=bool, default=False)
    parser.add_argument("--hyperplane_components", type=int, default=1)
    parser.add_argument("--p_normal_idx", type=str, default="uniform")
    parser.add_argument("--p_normal_value", type=str, default="uniform")
    parser.add_argument("--p_intercept", type=str, default="uniform")
    parser.add_argument("--prior_sample_size", type=float, default=0.1)
    parser.add_argument("--score_reduction", type=str, default="mean")
    parser.add_argument("--query_strategy", type=str, default="random")
    args = parser.parse_args()
    config = vars(args)

    comet_ml.init()
    for seed in range(32):
        experiment = comet_ml.OfflineExperiment(
            project_name="balif", offline_directory="comet"
        )
        experiment.log_parameters({"seed": seed, **config})

        for dataset_name in tqdm(odds_datasets.datasets_names, desc="datasets"):
            data, labels = odds_datasets.load(dataset_name)

            with experiment.context_manager(dataset_name):
                sim_results = run_fn(config, data, labels, jr.key(seed))

                for step, scores in enumerate(sim_results):
                    ap = avg_precision(labels, scores)
                    experiment.log_metric("average_precision", ap, step)
        experiment.end()
