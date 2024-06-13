
def get_score_matrix(max_depth, max_samples):
    def p(start, end):
        start, end = start - 2, end - 1
        bin_coef_ln = gammaln(start + 1) - gammaln(end + 1) - gammaln(start - end + 1)
        return jnp.exp(bin_coef_ln + start * jnp.log(0.5))

    # get the transition matrix for the Markov chain of splitting sizes
    M = jax.vmap(jax.vmap(p, in_axes=(0, None)), in_axes=(None, 0))(
        jnp.arange(1, max_samples + 1), jnp.arange(1, max_samples + 1)
    )
    M = M.at[:, 0].set(0).at[0, 0].set(1)

    # start with a prob vector with all mass in the last element (max node size)
    p = jnp.zeros(max_samples).at[-1].set(1)

    # compute the k-step transition matrix for k = 0, ..., hmax
    Mk = jax.lax.associative_scan(jnp.matmul, jnp.stack([M] * max_depth))
    Mk = jnp.concatenate([jnp.eye(max_samples)[jnp.newaxis], Mk], axis=0)

    # compute p after i steps for i = 0, ..., hmax
    # add zero so indexing is correct
    p = jax.vmap(jnp.matmul, in_axes=(0, None))(Mk, p)
    p = jnp.concatenate([jnp.zeros((max_depth + 1, 1)), p], axis=1)

    # compute the cumulative distribution function
    pc = p.cumsum(axis=1)
    pc = (pc.at[:, 1:].add(pc[:, :-1])) / 2  # assume points are "middle of the pack"
    pc = jnp.clip(pc, 0.01, 0.99).at[:, 0].set(0)  # fix floating point errors
    return pc



@classmethod
    @eqx.filter_jit
    def from_isolation_tree(
        cls,
        itree: IsolationTree,
        score_matrix: jax.Array,
        prior_sample_size: jnp.float_,
    ):
        def base_scores(itree: IsolationTree) -> jax.Array:
            def scan_score(_, idx):
                return _, 1 - score_matrix[idx[0], idx[1]]

            n_nodes, n_features = itree.normals.shape
            node_depths = jax.vmap(itree.depth)(jnp.arange(n_nodes))
            _, scores = jax.lax.scan(scan_score, None, (node_depths, itree.node_sizes))
            return scores

        def get_priors(itree: IsolationTree):
            """Compute the prior for each node in the tree"""
            scores = base_scores(itree)

            # match the predition adding strictly positive virtual samples
            sample_size_after_IF = (
                prior_sample_size  # / (jnp.minimum(scores, 1 - scores))
            )
            alphas = sample_size_after_IF * scores
            betas = sample_size_after_IF * (1 - scores)
            return alphas, betas

        alphas, betas = get_priors(itree)
        return cls(itree.normals, itree.intercepts, itree.node_sizes, alphas, betas)