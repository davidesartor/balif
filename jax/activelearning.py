@partial(jax.jit, static_argnums=(-1,))
def interest_for(self, point: jax.Array, strat="margin") -> jax.Array:
    if strat == "anom":
        return self.score(point)
    elif strat == "margin":
        alpha, beta = self.get_distr_params(point)
        # alpha beta sono le distr di ogni nodo, il margine è globale
        # da sistemare una volta decisa la regola di riduzione
        r = 0.5
        margin = jax.scipy.stats.beta.pdf(r, alpha, beta)
        return margin
    else:
        raise ValueError(f"Unknown interest strategy {strat}")
