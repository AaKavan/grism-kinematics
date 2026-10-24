"""Chunked NUTS-within-Gibbs with uninterrupted adaptation and exact restart.

This module owns no files and launches no processes. The calling worker records
settings, code/data hashes, and package versions before persisting ``state``.
Both warmup and retained draws advance the same public NumPyro kernel state.
"""
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import HMCGibbs, NUTS
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import unconstrain_fn


class RotationSampler:
    """Vectorized chains whose warmup can be checkpointed after any chunk.

    ``initial_values`` contains constrained physical values for every latent
    site, including the positive ``continuum_coefficients``. HMC coordinates
    are unconstrained here; Gibbs coordinates stay physical. On restart the
    same settings and initial values must accompany the saved NumPyro state.
    ``advance`` returns host arrays with leading [chain, iteration] dimensions.
    """
    gibbs_site = 'continuum_coefficients'

    def __init__(self, model, *, chains, warmup, target_accept=.9,
                 max_tree_depth=9, seed=718, initial_values, state=None):
        if int(chains) != chains or chains < 1 or len(initial_values) != chains:
            raise ValueError('One constrained initialization is required for each chain.')
        if int(warmup) != warmup or warmup < 0:
            raise ValueError('warmup must be a nonnegative integer.')
        if not .5 < target_accept < 1 or not 1 <= max_tree_depth <= 12:
            raise ValueError('Invalid target acceptance or maximum tree depth.')
        self.model = model
        self.chains, self.warmup = int(chains), int(warmup)
        self.seed = int(seed)
        self._advance_functions = {}
        initial = []
        expected_sites = set(initial_values[0])
        for values in initial_values:
            if set(values) != expected_sites or self.gibbs_site not in values:
                raise ValueError('Initial chains must contain the same sites, including continuum_coefficients.')
            if not all(np.isfinite(np.asarray(v)).all() for v in values.values()):
                raise ValueError('Initial physical values must be finite.')
            if np.any(np.asarray(values[self.gibbs_site]) <= 0):
                raise ValueError('Initial continuum coefficients must be strictly positive.')
            coordinates = unconstrain_fn(model.model, (), {}, values)
            coordinates[self.gibbs_site] = jnp.asarray(values[self.gibbs_site])
            if not all(np.isfinite(np.asarray(v)).all() for v in coordinates.values()):
                raise ValueError('Initial HMC values must lie strictly inside their prior bounds.')
            initial.append(coordinates)
        initial = jax.tree.map(lambda *v: jnp.stack(v), *initial)
        scalars = tuple(name for name in getattr(model, 'scalar_sites', ())
                        if name in expected_sites and name != self.gibbs_site)
        nuts = NUTS(model.model, init_strategy=init_to_value(values=initial_values[0]),
                    dense_mass=[scalars] if scalars else False,
                    target_accept_prob=target_accept, max_tree_depth=max_tree_depth)
        self.kernel = HMCGibbs(nuts, gibbs_fn=model.gibbs_fn, gibbs_sites=[self.gibbs_site])
        keys = jax.random.split(jax.random.key(self.seed), self.chains)
        # Calling init also constructs kernel transforms and conditional-potential
        # closures. It is required on restart even though its state is discarded.
        init_one = lambda key, coordinates: self.kernel.init(
            key, self.warmup, coordinates, (), {})
        initialized = jax.vmap(init_one)(keys, initial)
        jax.block_until_ready(initialized)
        self._state = initialized if state is None else self._validated_state(state, initialized)
        self._constrain = self.kernel.postprocess_fn((), {})
        self._sample_batch = jax.vmap(lambda s: self.kernel.sample(s, (), {}))
        # Check agreement here rather than discovering a damaged resume after a
        # costly compiled block. The worker verifies target/settings provenance.
        _ = self.iterations

    @staticmethod
    def _validated_state(state, initialized):
        if jax.tree.structure(state) != jax.tree.structure(initialized):
            raise ValueError('Saved sampler state has incompatible NumPyro structure.')
        for saved, reference in zip(jax.tree.leaves(state), jax.tree.leaves(initialized)):
            if np.shape(saved) != np.shape(reference) or saved.dtype != reference.dtype:
                raise ValueError('Saved sampler state has incompatible shapes or dtypes.')
        return jax.device_put(state)

    @property
    def state(self):
        """Current NumPyro state; persist ``jax.device_get(state)`` atomically."""
        return self._state

    @property
    def iterations(self):
        """Total transitions per chain, including warmup, without chunk resets."""
        counts = np.asarray(self._state.hmc_state.i)
        if counts.shape != (self.chains,) or np.any(counts != counts[0]) or counts[0] < 0:
            raise ValueError('Sampler chains have incompatible iteration counters.')
        return int(counts[0])

    @property
    def step_sizes(self):
        return np.asarray(self._state.hmc_state.adapt_state.step_size)

    def _compiled_advance(self, n, collect):
        key = (n, collect)
        if key not in self._advance_functions:
            def body(state, _):
                state = self._sample_batch(state)
                hmc = state.hmc_state
                stats = dict(energy=hmc.energy, potential_energy=hmc.potential_energy,
                    num_steps=hmc.num_steps, accept_prob=hmc.accept_prob,
                    diverging=hmc.diverging, iteration=hmc.i,
                    is_warmup=hmc.i <= self.warmup)
                samples = jax.vmap(self._constrain)(state.z) if collect else {}
                return state, (samples, stats)

            self._advance_functions[key] = jax.jit(
                lambda state: jax.lax.scan(body, state, xs=None, length=n))
        return self._advance_functions[key]

    def advance(self, n, collect=True):
        """Advance n global iterations; collect=False omits only sample storage.

        Warmup follows the fixed total specified at construction and stops
        automatically at that iteration. Collected warmup values, if requested,
        are marked by ``is_warmup`` and must not be used as posterior draws.
        Sampling chunks never restart adaptation or reseed a chain.
        """
        if int(n) != n or n < 1:
            raise ValueError('A positive integer number of transitions is required.')
        before = self.iterations
        state, (samples, stats) = self._compiled_advance(int(n), bool(collect))(self._state)
        jax.block_until_ready(state)
        # Save the successfully completed state before transferring draw arrays.
        self._state = state
        if self.iterations != before+n:
            raise RuntimeError('The sampler iteration counter did not advance as requested.')
        to_host = lambda x: np.asarray(x).swapaxes(0, 1)
        return jax.tree.map(to_host, samples), jax.tree.map(to_host, stats)
