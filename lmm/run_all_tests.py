"""
Master Test Suite for Project Buddha LMM.
Verifies all 6 components work end-to-end.
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp

from lmm.config import LMMConfig
from lmm.ast_graph import MathAST, NodeType
from lmm.neural_brain import init_brain
from lmm.fitness import make_gravity_dataset, make_spring_dataset, evaluate_fitness
from lmm.mcts import MCTS
from lmm.trainer import LMMTrainer
from lmm.inference import DiscoveryEngine


def test_suite():
    print("=" * 80)
    print("      PROJECT BUDDHA: LMM MASTER VERIFICATION SUITE")
    print("=" * 80)

    cfg = LMMConfig(max_vars=2)
    key = jax.random.PRNGKey(42)

    # 1. AST Graph Test
    print("\n[1/6] Testing AST Graph Engine...")
    ast = MathAST(max_nodes=cfg.max_nodes, num_vars=cfg.max_vars)
    ast = ast.apply_action(int(NodeType.NEG))
    ast = ast.apply_action(int(NodeType.VAR_0))
    assert ast.is_complete
    assert ast.to_string(["x", "v"]) == "(-x)"
    feat, adj = ast.to_tensors()
    assert feat.shape == (cfg.max_nodes, cfg.vocab_size)
    assert adj.shape == (cfg.max_nodes, cfg.max_nodes)
    print("  ✓ AST Graph Engine verified.")

    # 2. Neural Brain Test
    print("\n[2/6] Testing MPNN Neural Brain...")
    model, params = init_brain(cfg, key)
    feat_b = jnp.expand_dims(jnp.array(feat), axis=0)
    adj_b = jnp.expand_dims(jnp.array(adj), axis=0)
    policy_logits, value = model.apply(params, feat_b, adj_b)
    assert policy_logits.shape == (1, cfg.action_space_size)
    assert value.shape == (1, 1)
    assert -1.0 <= float(value[0, 0]) <= 1.0
    print("  ✓ MPNN Neural Brain verified.")

    # 3. Fitness Evaluator Test
    print("\n[3/6] Testing Fitness Evaluator (RK4)...")
    spring_ds = make_spring_dataset()
    fitness = evaluate_fitness(ast, spring_ds, cfg)
    assert fitness > 0.95, f"Spring fitness was {fitness}, expected > 0.95"
    print(f"  ✓ Fitness Evaluator verified: {ast.to_string()} on spring scores {fitness:.4f}")

    # 4. MCTS Engine Test
    print("\n[4/6] Testing MCTS PUCT Search...")
    mcts = MCTS(model, params, cfg)
    root_ast = MathAST(max_nodes=10, num_vars=2)
    pi_dict, root_node = mcts.search(root_ast, num_simulations=20)
    assert len(pi_dict) > 0
    assert abs(sum(pi_dict.values()) - 1.0) < 1e-4
    print("  ✓ MCTS PUCT Search verified.")

    # 5. Trainer & Backprop Test
    print("\n[5/6] Testing Self-Play Trainer...")
    trainer = LMMTrainer(cfg, spring_ds, seed=1)
    trainer.train_episodes(num_episodes=2, print_interval=1)
    assert len(trainer.buffer) > 0
    print("  ✓ Self-Play Trainer verified.")

    # 6. Autonomous Discovery Engine
    print("\n[6/6] Testing Discovery Engine Interface...")
    engine = DiscoveryEngine(cfg, model=trainer.model, params=trainer.params)
    best_ast, best_score, eq_str = engine.discover(spring_ds, num_attempts=2, simulations_per_move=20)
    assert best_ast is not None
    print(f"  ✓ Discovery Engine verified: Discovered '{eq_str}' with fitness {best_score:.4f}")

    print("\n" + "=" * 80)
    print("      ALL 6 LMM COMPONENTS VERIFIED & READY FOR SCIENTIFIC DISCOVERY ✓")
    print("=" * 80)


if __name__ == "__main__":
    test_suite()
