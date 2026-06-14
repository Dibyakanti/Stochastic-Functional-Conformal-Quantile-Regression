from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax


class LangevinNoiseState(NamedTuple):
    rng_key: jax.Array


def add_langevin_noise(
    learning_rate: float,
    temperature: float = 1.0,
    seed: int = 0,
):
    """Add Langevin noise to optimizer updates.

    Update becomes

        u <- u + sqrt(2 * lr * temperature) * N(0, I)

    where u is the update produced by previous Optax transforms.
    """

    sigma = jnp.sqrt(2.0 * learning_rate * temperature)

    def init_fn(params):
        return LangevinNoiseState(
            rng_key=jax.random.PRNGKey(seed)
        )

    def update_fn(updates, state, params=None):
        del params

        leaves, treedef = jax.tree_util.tree_flatten(updates)

        keys = jax.random.split(state.rng_key, len(leaves) + 1)
        new_key = keys[0]
        noise_keys = keys[1:]

        noisy_leaves = [
            u + sigma * jax.random.normal(k, u.shape, u.dtype)
            for u, k in zip(leaves, noise_keys)
        ]

        noisy_updates = jax.tree_util.tree_unflatten(
            treedef, noisy_leaves
        )

        return noisy_updates, LangevinNoiseState(
            rng_key=new_key
        )

    return optax.GradientTransformation(
        init_fn,
        update_fn,
    )  

class ASGLDState(NamedTuple):
    count: jnp.ndarray
    m: optax.Updates
    v: optax.Updates
    rng_key: jax.Array
    
def asgld(
    learning_rate,
    beta1=0.9,
    beta2=0.999,
    a=1.0,
    tau=1.0,
    eps=1e-8,
):
    def init_fn(params):
        return ASGLDState(
            count=jnp.zeros([], jnp.int32),
            m=jax.tree.map(jnp.zeros_like, params),
            v=jax.tree.map(jnp.zeros_like, params),
            rng_key=jax.random.PRNGKey(0),
        )

    def update_fn(grads, state, params=None):
        del params

        m = jax.tree.map(
            lambda m, g: beta1 * m + (1.0 - beta1) * g,
            state.m,
            grads,
        )

        v = jax.tree.map(
            lambda v, g: beta2 * v + (1.0 - beta2) * g**2,
            state.v,
            grads,
        )

        lr = learning_rate

        drift = jax.tree.map(
            lambda g, m_, v_: (
                -lr * (
                    g
                    + a * m_ / jnp.sqrt(v_ + eps)
                )
            ),
            grads,
            m,
            v,
        )

        key, subkey = jax.random.split(state.rng_key)

        leaves, treedef = jax.tree.flatten(drift)
        keys = jax.random.split(subkey, len(leaves))

        noise_leaves = [
            jnp.sqrt(2.0 * tau * lr)
            * jax.random.normal(k, x.shape, x.dtype)
            for k, x in zip(keys, leaves)
        ]

        noise = jax.tree.unflatten(treedef, noise_leaves)

        updates = jax.tree.map(
            lambda d, n: d + n,
            drift,
            noise,
        )

        new_state = ASGLDState(
            count=state.count + 1,
            m=m,
            v=v,
            rng_key=key,
        )

        return updates, new_state

    return optax.GradientTransformation(
        init_fn,
        update_fn,
    )


class SgldAdamState(NamedTuple):
    count: jnp.ndarray       # Iteration k
    m: optax.Updates         # First moment m_k
    v: optax.Updates         # Second moment v_k
    rng: jax.random.PRNGKey  # RNG key for Langevin noise injection

def adam_sgld(
    learning_rate, 
    rng_key: jax.random.PRNGKey = jax.random.PRNGKey(42),
    b1: float = 0.9, 
    b2: float = 0.999, 
    eps: float = 1e-8,
    beta: float = 1.0
) -> optax.GradientTransformation:
    """Implements Adam-preconditioned Stochastic Gradient Langevin Dynamics."""
    
    def init_fn(params) -> SgldAdamState:
        # Initialize m_0 = 0, v_0 = 0, k = 0
        m = jax.tree_util.tree_map(jnp.zeros_like, params)
        v = jax.tree_util.tree_map(jnp.zeros_like, params)
        return SgldAdamState(count=jnp.zeros([], dtype=jnp.int32), m=m, v=v, rng=rng_key)

    def update_fn(updates, state, params=None):
        # 1. Increment iteration count (k + 1)
        k_next = state.count + 1
        
        # Split RNG key for current step's Langevin noise
        rng_curr, rng_next = jax.random.split(state.rng)
        
        # 2. Update biased first moment: m_{k+1} = b1 * m_k + (1 - b1) * g_k
        m_next = jax.tree_util.tree_map(
            lambda m_g, g: b1 * m_g + (1.0 - b1) * g, state.m, updates
        )
        
        # 3. Update biased second moment: v_{k+1} = b2 * v_k + (1 - b2) * g_k^2
        v_next = jax.tree_util.tree_map(
            lambda v_g, g: b2 * v_g + (1.0 - b2) * jnp.square(g), state.v, updates
        )
        
        # 4. Bias-correct moments
        # m_hat = m_{k+1} / (1 - b1^{k+1})
        # v_hat = v_{k+1} / (1 - b2^{k+1})
        m_hat = jax.tree_util.tree_map(lambda m_g: m_g / (1.0 - b1**k_next), m_next)
        v_hat = jax.tree_util.tree_map(lambda v_g: v_g / (1.0 - b2**k_next), v_next)
        
        # Get learning rate for the current step: eta_k
        if callable(learning_rate):
            eta = learning_rate(state.count)
        else:
            eta = learning_rate

        # Generate standard normal noise streams matching pytree structure
        # xi_k ~ N(0, I)
        treedef = jax.tree_util.tree_structure(updates)
        keys = jax.random.split(rng_curr, treedef.num_leaves)
        xi = jax.tree_util.tree_unflatten(treedef, [jax.random.normal(k, p.shape) for k, p in zip(keys, jax.tree_util.tree_leaves(updates))])
        
        # 5 & 6. Preconditioner Diagonal, Langevin Noise Injection, and Final Parameter Step
        # P_{k+1} = 1 / (sqrt(v_hat) + eps)
        # step = - eta * P_{k+1} * m_hat + sqrt((2 * eta) / beta) * P_{k+1}^{1/2} * xi
        def compute_step(m_h, v_h, noise):
            p_diag = 1.0 / (jnp.sqrt(v_h) + eps)                    # P_{k+1}
            p_half = jnp.sqrt(p_diag)                               # P_{k+1}^{1/2}
            
            deterministic_term = - eta * p_diag * m_h
            stochastic_scale = jnp.sqrt((2.0 * eta) / beta)
            stochastic_term = stochastic_scale * p_half * noise
            
            return deterministic_term + stochastic_term

        new_updates = jax.tree_util.tree_map(compute_step, m_hat, v_hat, xi)
        
        new_state = SgldAdamState(count=k_next, m=m_next, v=v_next, rng=rng_next)
        return new_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)