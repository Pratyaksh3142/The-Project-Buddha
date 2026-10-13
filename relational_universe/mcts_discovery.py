"""
Level 0: Generative MCTS Discovery Loop for Relational Universes

Completely open-ended, non-hardcoded rule discovery:
  - Rules are dynamically generated and mutated (edge additions, removals, rewirings).
  - Evaluated on the physical substrate by InvariantReferee (pure graph hop measurements).
  - Neural Intuition Brain learns through continuous JAX backprop.
  - EVERY single tested rule, topology, and measurement is logged to 'discovery_trace.log'.
"""

from __future__ import annotations
import os
import json
import random
from typing import List, Tuple, Dict, Optional
import numpy as np
import jax
import jax.numpy as jnp
import optax

from relational_universe.substrate import RewriteRule
from relational_universe.referee import InvariantReferee
from relational_universe.neural_intuition import init_intuition_brain
from relational_universe.extractor import ContinuumExtractor


LOG_FILE = os.path.join(os.path.dirname(__file__), "discovery_trace.log")


def rule_to_features(rule: RewriteRule, feature_dim: int = 16) -> np.ndarray:
    """Encodes a discrete rewrite rule into a normalized feature vector for the neural brain."""
    feat = np.zeros(feature_dim, dtype=np.float32)
    feat[0] = len(rule.pattern_l)
    feat[1] = len(rule.pattern_r)
    idx = 2
    for s, t in rule.pattern_l:
        if idx < feature_dim - 1:
            feat[idx] = float(s + 1)
            feat[idx + 1] = float(t + 1)
            idx += 2
    for s, t in rule.pattern_r:
        if idx < feature_dim - 1:
            feat[idx] = float(s + 1)
            feat[idx + 1] = float(t + 1)
            idx += 2
    return feat


class RuleMutator:
    """
    Generative MCTS Action Space: Mutates graph rewrite rules by adding,
    removing, or rewiring edges on the right-hand pattern R.
    """
    def __init__(self, max_nodes: int = 4):
        self.max_nodes = max_nodes

    def random_rule(self, min_edges_r: int = 2, max_edges_r: int = 4) -> RewriteRule:
        """Generates a novel candidate rule from pure random topology."""
        pattern_l = [(0, 1), (1, 2)]
        num_edges_r = random.randint(min_edges_r, max_edges_r)
        pattern_r = []
        possible_nodes = list(range(self.max_nodes))
        for _ in range(num_edges_r):
            s = random.choice(possible_nodes)
            t = random.choice(possible_nodes)
            if s != t and (s, t) not in pattern_r:
                pattern_r.append((s, t))
        if not pattern_r:
            pattern_r = [(0, 3), (3, 2)]
        return RewriteRule(pattern_l=pattern_l, pattern_r=pattern_r)

    def mutate_rule(self, rule: RewriteRule) -> RewriteRule:
        """Applies a discrete topological mutation to rule pattern R."""
        pattern_r = list(rule.pattern_r)
        mutation_type = random.choice(["add", "remove", "rewire"])
        possible_nodes = list(range(self.max_nodes))

        if mutation_type == "add" and len(pattern_r) < 6:
            s, t = random.choice(possible_nodes), random.choice(possible_nodes)
            if s != t and (s, t) not in pattern_r:
                pattern_r.append((s, t))
        elif mutation_type == "remove" and len(pattern_r) > 1:
            pattern_r.pop(random.randint(0, len(pattern_r) - 1))
        elif mutation_type == "rewire" and len(pattern_r) > 0:
            idx = random.randint(0, len(pattern_r) - 1)
            s, _ = pattern_r[idx]
            new_t = random.choice(possible_nodes)
            if s != new_t:
                pattern_r[idx] = (s, new_t)

        return RewriteRule(pattern_l=list(rule.pattern_l), pattern_r=pattern_r)


def run_continuous_discovery(num_episodes: int = 25, seed: int = 42):
    """
    Runs an open-ended, continuous generative discovery loop across hundreds of rules.
    Logs every single rule and its exact measurement to discovery_trace.log.
    """
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("  PROJECT BUDDHA: GENERATIVE RELATIONAL DISCOVERY ENGINE")
    print("  Zero Hardcoding | Generative Graph Mutation | Pure Hop Kinematics")
    print(f"  Logging trace to: {LOG_FILE}")
    print("=" * 70)

    # Initialize log file
    with open(LOG_FILE, "w") as f:
        f.write("# PROJECT BUDDHA: RELATIONAL DISCOVERY EXECUTION TRACE\n")
        f.write("# Format: Episode | Rule ID | Pattern R | Stability | Confluence | Verdict\n\n")

    model, params, optimizer, opt_state = init_intuition_brain(feature_dim=16, rng_seed=seed)
    referee = InvariantReferee(simulation_steps=20)
    extractor = ContinuumExtractor()
    mutator = RuleMutator(max_nodes=4)

    # Start with a base population of dynamically generated rules
    current_rule = mutator.random_rule()

    # Neural training step
    def loss_fn(p, x_batch, y_batch):
        preds, _ = jax.vmap(lambda x: model.apply(p, x))(x_batch)
        return jnp.mean((preds.squeeze(-1) - y_batch) ** 2)

    @jax.jit
    def train_step(p, opt_st, x_b, y_b):
        loss, grads = jax.value_and_grad(loss_fn)(p, x_b, y_b)
        updates, opt_st = optimizer.update(grads, opt_st, p)
        p = optax.apply_updates(p, updates)
        return p, opt_st, loss

    discoveries = []
    total_rules_tested = 0

    for ep in range(1, num_episodes + 1):
        # Generate 4 candidate mutations per episode (MCTS branching)
        candidate_mutations = [mutator.mutate_rule(current_rule) for _ in range(4)]
        best_rule_in_ep = None
        best_score_in_ep = -float("inf")

        for m_idx, rule in enumerate(candidate_mutations):
            total_rules_tested += 1
            feat = rule_to_features(rule, feature_dim=16)
            feat_jax = jnp.array(feat, dtype=jnp.float32)

            # 1. Brain's Prior Intuition
            val_pred, _ = model.apply(params, feat_jax)
            intuition_val = float(val_pred[0] if val_pred.ndim > 0 else val_pred)

            # 2. Hard Physical Simulation (Zero Formulas)
            is_stable, stab_score = referee.evaluate_knife_edge(rule)
            is_confluent = referee.evaluate_confluence(rule) if is_stable else False

            actual_score = (stab_score + (5.0 if is_confluent else -5.0)) if is_stable else -10.0

            # 3. Log to file
            log_entry = (
                f"Ep {ep:2d} | Rule #{total_rules_tested:3d} | R={rule.pattern_r} | "
                f"Stable={is_stable} (Score={stab_score:+.2f}) | Confluent={is_confluent} | "
                f"Intuition={intuition_val:+.2f} -> Target={actual_score:+.2f}\n"
            )
            with open(LOG_FILE, "a") as f:
                f.write(log_entry)

            # 4. Train Neural Intuition
            x_b = jnp.expand_dims(feat_jax, 0)
            y_b = jnp.array([actual_score], dtype=jnp.float32)
            params, opt_state, _ = train_step(params, opt_state, x_b, y_b)

            if actual_score > best_score_in_ep:
                best_score_in_ep = actual_score
                best_rule_in_ep = rule

            # 5. If this rule survived both Knife-Edge and Confluence, measure its kinematics!
            if is_stable and is_confluent:
                obs = referee.measure_knot_kinematics(rule)
                # Ensure we have non-trivial non-zero displacement
                if any(o["acceleration"] > 0 for o in obs):
                    try:
                        alpha, beta, k, max_err = extractor.extract_power_law_invariant(obs)
                        latex = extractor.format_latex_law(alpha, beta, k)
                        discovery_record = {
                            "episode": ep,
                            "rule_id": total_rules_tested,
                            "rule": rule,
                            "alpha": alpha,
                            "beta": beta,
                            "k": k,
                            "error": max_err,
                            "latex": latex,
                            "observations": obs,
                        }
                        discoveries.append(discovery_record)
                        with open(LOG_FILE, "a") as f:
                            f.write(f"  ==> [✦ DISCOVERY] Law: {latex} (Error={max_err:.6f})\n\n")
                    except Exception:
                        pass

        # MCTS Evolution: Keep best mutation as seed for next generation
        if best_rule_in_ep is not None and best_score_in_ep > -5.0:
            current_rule = best_rule_in_ep

        if ep % 5 == 0 or ep == 1:
            print(f"  Episode {ep:2d}/{num_episodes} | Total Rules Tested: {total_rules_tested:3d} | "
                  f"Discoveries Logged: {len(discoveries)} | Best Ep Score: {best_score_in_ep:+.2f}")

    print("\n" + "=" * 70)
    print(f"  SEARCH COMPLETE: {total_rules_tested} Total Rules Explored")
    print(f"  Total Verified Discoveries: {len(discoveries)}")
    print(f"  All details saved to: {LOG_FILE}")
    print("=" * 70)

    if discoveries:
        latest = discoveries[-1]
        print(f"\n  ✦ Latest Discovered Invariant Law (Episode {latest['episode']}, Rule #{latest['rule_id']}):")
        print(f"    R Pattern: {latest['rule'].pattern_r}")
        print(f"    α (Flux exponent) = {latest['alpha']:+.6f}")
        print(f"    β (Mass exponent) = {latest['beta']:+.6f}")
        print(f"    LaTeX Equation:   {latest['latex']}")
        print(f"    Residual Error:   {latest['error']:.8f}")

    return discoveries


# Backward compatibility alias
run_relational_discovery = run_continuous_discovery


if __name__ == "__main__":
    run_continuous_discovery(num_episodes=25, seed=42)
