import jax.numpy as jnp

from train import trainers

def autoregressive_unroll(trainer, start_state, start_time, num_steps, dt):
    """
    Predicts a sequence of future states by feeding predictions back into the model.
    """
    variables = trainer.train_state.model_variables
    current_state = start_state
    trajectory = [start_state]
    
    # We use the inference function logic from the model
    for i in range(num_steps):
        # Update time for each step: t_next = t_now + dt
        current_time = start_time + (i * dt)
        
        # Predict the next field
        next_state = trainer.model.architecture.apply(
            variables,
            current_state,
            sigma=jnp.array([0.0]),  # Assuming zero noise for pure prediction
            time=jnp.array([current_time]),
            is_training=False
        )
        
        trajectory.append(next_state)
        current_state = next_state
        
    return jnp.concatenate(trajectory, axis=0)

if __name__=="__main__":