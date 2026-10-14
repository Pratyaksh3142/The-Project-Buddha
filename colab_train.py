"""
Colab & Kaggle Training Launcher for Project Buddha Unified Discovery Engine.

Instructions for Google Colab / Kaggle:
1. Clone repo:
   !git clone https://github.com/Pratyaksh3142/The-Project-Buddha.git
   %cd The-Project-Buddha
2. Install dependencies:
   !pip install -q jax jaxlib flax optax numpy
3. Run this script:
   !python3 colab_train.py --hours 4.0 --dims 4 5 6 --channels 6 --sims 64
"""

import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_universe.train_unified import run_training
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colab/Kaggle Unified Discovery Trainer")
    parser.add_argument("--hours", type=float, default=2.0, help="Training duration in hours")
    parser.add_argument("--dims", type=int, nargs="+", default=[4, 5, 6], help="Entity curriculum N")
    parser.add_argument("--channels", type=int, default=6, help="Relational channels K")
    parser.add_argument("--sims", type=int, default=64, help="MCTS simulations per step")
    parser.add_argument("--steps", type=int, default=8, help="Max steps per episode")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--model-dim", type=int, default=64, help="Pairformer dimension")
    parser.add_argument("--layers", type=int, default=3, help="Pairformer layers")
    parser.add_argument("--output", type=str, default="./discovery_output", help="Output directory")

    args = parser.parse_args()
    print("=" * 80)
    print("  PROJECT BUDDHA: HIGH-THROUGHPUT DISCOVERY RUNNER (Colab/Kaggle/M2)")
    print("=" * 80)
    run_training(
        duration_hours=args.hours,
        entity_curriculum=args.dims,
        num_channels=args.channels,
        sims_per_step=args.sims,
        max_steps=args.steps,
        batch_size=args.batch,
        dim=args.model_dim,
        num_layers=args.layers,
        output_dir=args.output,
    )
