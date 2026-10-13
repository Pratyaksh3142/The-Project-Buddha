"""
End-to-end experiment: Train the LMM to discover Newtonian Gravity (dv/dt = -1/r^2).
"""

from lmm.config import LMMConfig
from lmm.fitness import make_gravity_dataset
from lmm.trainer import LMMTrainer
from lmm.inference import DiscoveryEngine


def main():
    print("=" * 80)
    print("      PROJECT BUDDHA: END-TO-END DISCOVERY EXPERIMENT")
    print("      TARGET LAW: NEWTONIAN GRAVITY (dv/dt = -1/r^2)")
    print("=" * 80)

    cfg = LMMConfig(
        max_nodes=14,
        max_vars=2,
        embedding_dim=128,
        mpnn_layers=3,
        mcts_simulations=40,
        batch_size=8,
        learning_rate=2e-3
    )

    dataset = make_gravity_dataset()
    print(f"Dataset generated: {dataset.name} with {len(dataset.trajectories)} test trajectories.")

    # 1. Train the Brain through Self-Play RL
    trainer = LMMTrainer(cfg, dataset, seed=314)
    trainer.train_episodes(num_episodes=25, print_interval=5)

    # 2. Run Inference using the trained brain
    print("\nStarting Guided Discovery with Trained Brain...")
    engine = DiscoveryEngine(cfg, model=trainer.model, params=trainer.params)
    best_ast, best_fitness, best_eq = engine.discover(dataset, num_attempts=10, simulations_per_move=80)

    print("\n" + "=" * 80)
    print(f"FINAL RESULT:")
    print(f"  Target Law:        dv/dt = -1 / r^2")
    print(f"  Discovered Formula: dv/dt = {best_eq}")
    print(f"  Fitness Score:     {best_fitness:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
