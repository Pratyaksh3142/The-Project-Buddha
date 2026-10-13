#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import optax
import numpy as np

# Import our Buddha Components
from neural_alphamath import initialize_model, ACTION_SPACE_SIZE
from mcts_alphamath import GraphMCTS
from graph_engine import ComputationGraph
from autonomous_discovery import PhysicalDataset, evaluate_fitness
from rl_train_alphamath import RLComputationGraph, ReplayBuffer, train_step_jit

def make_gravity_dataset() -> PhysicalDataset:
    """
    Simulates an object falling in a 1/r^2 gravitational field.
    dr/dt = v
    dv/dt = -1 / r^2
    """
    dimension = 2
    var_names = ['r', 'v']
    t_end = 1.0  # Simulate for 1 second (stops before hitting r=0 singularity)
    dt = 0.05

    def true_derivs(y):
        r, v = y[0], y[1]
        return [v, -1.0 / (r**2)]

    def gen_traj(y0):
        y = list(y0)
        t = 0.0
        traj = [list(y)]
        while t < t_end - 1e-12:
            k1 = true_derivs(y)
            y2 = [y[i] + 0.5 * dt * k1[i] for i in range(2)]
            k2 = true_derivs(y2)
            y3 = [y[i] + 0.5 * dt * k2[i] for i in range(2)]
            k3 = true_derivs(y3)
            y4 = [y[i] + dt * k3[i] for i in range(2)]
            k4 = true_derivs(y4)
            for i in range(2):
                y[i] += (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
            t += dt
            traj.append(list(y))
        return traj

    # 3 independent falling trajectories
    test_ics = [
        [2.0, 0.0],   # Dropped from distance 2
        [3.0, -0.5],  # Thrown inward from distance 3
        [1.5, 0.5],   # Thrown outward from distance 1.5
    ]
    ic_trajs = [(ic, gen_traj(ic)) for ic in test_ics]

    return PhysicalDataset(
        name="Newtonian Gravity (Inverse Square Law)",
        dimension=dimension,
        var_names=var_names,
        t_end=t_end,
        dt=dt,
        ic_trajectories=ic_trajs
    )

def run_gravity_experiment():
    print("================================================================================")
    print("      PROJECT BUDDHA: DISCOVERING THE LAW OF GRAVITY             ")
    print("================================================================================")
    
    dataset = make_gravity_dataset()
    print(f" Dataset loaded: {dataset.name}")
    print(" Goal: Engine must discover `dv/dt = -1 / r^2` from a blank slate.")
    
    # Initialize Brain
    key = jax.random.PRNGKey(99)
    model, params = initialize_model(key)
    optimizer = optax.adam(learning_rate=2e-3)
    opt_state = optimizer.init(params)
    buffer = ReplayBuffer()
    
    # Initialize MCTS
    class RLGraphMCTS(GraphMCTS):
        def _get_masked_probabilities(self, logits, legal_actions):
            probs = jax.nn.softmax(logits).squeeze()
            return {a: float(probs[a]) for a in legal_actions}
            
    mcts = RLGraphMCTS(model, params, num_simulations=15)
    
    NUM_EPISODES = 5
    for episode in range(NUM_EPISODES):
        print(f"\n--- Episode {episode+1}/{NUM_EPISODES} ---")
        cg = ComputationGraph(dimension=dataset.dimension)
        state = RLComputationGraph(cg)
        episode_memory_indices = []
        
        # Build the equation graph
        while not state.is_terminal():
            pi_dict = mcts.search(state)
            
            # Convert policy to array
            pi_arr = np.zeros(ACTION_SPACE_SIZE)
            for a, p in pi_dict.items(): pi_arr[a] = p
                
            # Sample action
            action = np.random.choice(list(pi_dict.keys()), p=list(pi_dict.values()))
            
            # Store in buffer
            feat, adj = state.to_jax_tensors()
            buffer.add(feat, adj, pi_arr, 0.0)
            episode_memory_indices.append(len(buffer.buffer) - 1)
            
            state = state.copy_and_apply(action)
            
        # Referee Evaluation
        fitness = evaluate_fitness(state.graph, dataset)
        eq_str = state.graph.to_string(dataset.var_names)
        if len(eq_str) > 60: eq_str = eq_str[:57] + "..."
        
        print(f" Proposed Law: {eq_str}")
        print(f" Fitness Score: {fitness:.6f}")
        
        # Backpropagate
        buffer.update_episode_values(episode_memory_indices, fitness)
        
        # Train Brain
        if len(buffer.buffer) >= 4:
            import random
            batch = random.sample(buffer.buffer, 4)
            b_feat = jnp.concatenate([b[0] for b in batch])
            b_adj = jnp.concatenate([b[1] for b in batch])
            b_pi = jnp.array([b[2] for b in batch])
            b_val = jnp.array([b[3] for b in batch])
            
            params, opt_state, loss = train_step_jit(params, opt_state, model, optimizer, b_feat, b_adj, b_pi, b_val)
            print(f" Brain updated! Network Loss: {loss:.4f}")
            mcts.params = params

if __name__ == "__main__":
    run_gravity_experiment()
