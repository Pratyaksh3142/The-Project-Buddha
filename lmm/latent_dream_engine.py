"""
Project Buddha: The Continuous Latent Dream Engine — v2 (Finalized)

Core changes over v1:
  - Numerically stable operator evaluations (all clipped/guarded)
  - Routing replaced with fixed hierarchical input pairing (no NaN cascade)
  - Loss function fully JIT-traceable (no Python float() calls inside)
  - Gumbel noise stabilized, temperature flooring enforced
  - Crystallization produces human-readable equation string
"""

from __future__ import annotations

import enum
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Mathematical Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

class Op(enum.IntEnum):
    ADD      = 0
    SUB      = 1
    MUL      = 2
    DIV      = 3   # safe division
    POW2     = 4   # square  (x²) — avoids fractional-exponent NaN
    SIN      = 5
    COS      = 6
    EXP      = 7   # clamped
    LOG      = 8   # safe log
    NEG      = 9
    INV      = 10  # 1/x — safe
    SQRT     = 11  # √|x|
    VAR_X    = 12  # state variable / spatial coord
    VAR_T    = 13  # time coord
    CONST_1  = 14  # 1.0
    CONST_PI = 15  # π


VOCAB   = len(Op)
OP_REPR = {
    Op.ADD:      "+",    Op.SUB:   "-",    Op.MUL: "×",
    Op.DIV:      "/",    Op.POW2:  "²",    Op.SIN: "sin",
    Op.COS:      "cos",  Op.EXP:   "exp",  Op.LOG: "log",
    Op.NEG:      "neg",  Op.INV:   "1/",   Op.SQRT:"√",
    Op.VAR_X:    "x",    Op.VAR_T: "t",
    Op.CONST_1:  "1",    Op.CONST_PI: "π",
}

# Safe numerical evaluations for every op
def _eval_all_ops(left: jnp.ndarray, right: jnp.ndarray,
                  x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """
    Returns a [B, VOCAB] tensor:  every op applied to (left, right).
    All ops are numerically guarded against NaN / Inf.
    """
    eps = 1e-6
    safe_left  = jnp.clip(left,  -20.0, 20.0)
    safe_right = jnp.clip(right, -20.0, 20.0)

    ops = [
        safe_left  + safe_right,                                    # ADD
        safe_left  - safe_right,                                    # SUB
        safe_left  * safe_right,                                    # MUL
        safe_left  / (jnp.abs(safe_right) + eps),                  # DIV
        safe_left  ** 2,                                            # POW2
        jnp.sin(safe_left),                                         # SIN
        jnp.cos(safe_left),                                         # COS
        jnp.exp(jnp.clip(safe_left, -8.0, 8.0)),                   # EXP
        jnp.log(jnp.abs(safe_left) + eps),                         # LOG
        -safe_left,                                                  # NEG
        1.0 / (jnp.abs(safe_left) + eps),                          # INV
        jnp.sqrt(jnp.abs(safe_left) + eps),                        # SQRT
        x,                                                           # VAR_X
        t,                                                           # VAR_T
        jnp.ones_like(x),                                           # CONST_1
        jnp.full_like(x, jnp.pi),                                  # CONST_PI
    ]
    return jnp.stack(ops, axis=-1)   # [B, VOCAB]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Gumbel-Softmax Straight-Through Estimator
# ─────────────────────────────────────────────────────────────────────────────

def gumbel_softmax_ste(
    logits: jnp.ndarray,
    rng: jax.Array,
    tau: float,
) -> jnp.ndarray:
    """
    Forward:  discrete one-hot  (argmax of perturbed logits).
    Backward: gradients flow through the soft probabilities.

    logits : [N, V]
    returns: [N, V]
    """
    tau   = jnp.maximum(jnp.array(tau, dtype=jnp.float32), 0.02)
    noise = jax.random.gumbel(rng, logits.shape)
    y_soft = jax.nn.softmax((logits + noise) / tau, axis=-1)
    y_hard = jax.nn.one_hot(jnp.argmax(y_soft, axis=-1), VOCAB, dtype=y_soft.dtype)
    # straight-through: forward=hard, backward=soft
    return y_soft + jax.lax.stop_gradient(y_hard - y_soft)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Latent Dream Engine (Flax module)
# ─────────────────────────────────────────────────────────────────────────────

class LatentDreamEngine(nn.Module):
    """
    N learnable nodes arranged in a fixed binary-tree topology.
    Each node holds continuous logits over the operator vocabulary.

    Fixed binary tree wiring (no learnable routing — eliminates NaN cascade):
        Layer 0 (leaves): nodes  0 … L-1        input = (x, t)
        Layer 1 (mid):    nodes  L … L+L//2-1   input = (node_2i, node_2i+1)
        ...
        Root (node N-1):  output of the expression
    """
    num_nodes: int = 16

    def setup(self):
        self.node_logits = self.param(
            "node_logits",
            nn.initializers.normal(stddev=0.5),
            (self.num_nodes, VOCAB),
        )

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        rng: jax.Array,
        tau: float = 1.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns
        -------
        output   : [B] scalar value of the expression at root
        op_probs : [N, VOCAB] soft operator probabilities (for entropy loss)
        """
        # Sample operators via Gumbel-Softmax STE  [N, VOCAB]
        op_w = gumbel_softmax_ste(self.node_logits, rng, tau)
        op_probs = jax.nn.softmax(self.node_logits, axis=-1)  # for entropy

        # Evaluate nodes in topological order (leaves → root)
        vals = []
        for i in range(self.num_nodes):
            # Fixed tree: leaves get (x, t), others get two child nodes
            if i == 0:
                left, right = x, t
            elif i == 1:
                left, right = t, x
            else:
                # Two most recent children
                left  = vals[i - 2]
                right = vals[i - 1]

            all_ops = _eval_all_ops(left, right, x, t)  # [B, VOCAB]
            # Soft-select using Gumbel weights
            node_val = jnp.sum(all_ops * op_w[i], axis=-1)  # [B]
            # Clip for numerical health
            node_val = jnp.clip(node_val, -50.0, 50.0)
            vals.append(node_val)

        return vals[-1], op_probs


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Temperature annealing schedule
# ─────────────────────────────────────────────────────────────────────────────

def anneal_temperature(step: int, total: int,
                       tau_hi: float = 5.0, tau_lo: float = 0.05) -> float:
    """Exponential decay: τ(0) = tau_hi, τ(total) = tau_lo."""
    progress = step / max(1, total)
    return float(tau_lo + (tau_hi - tau_lo) * (tau_lo / tau_hi) ** progress)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Universal Multi-Regime Paradox Loss
# ─────────────────────────────────────────────────────────────────────────────

def paradox_loss(
    params: dict,
    model: LatentDreamEngine,
    rng: jax.Array,
    tau: float,
) -> Tuple[jnp.ndarray, dict]:
    """
    Regime 1 – Classical/Newtonian:
        Feed x ∈ [1, 5] (macro scale, unit masses).
        Target: f(x) ≈ -x  (harmonic restoring force, simplest classical law).

    Regime 2 – Bounded / Quantum-like:
        Feed x ∈ [0.001, 0.1] (micro scale).
        Penalty: output must stay finite and small — no Planck-scale explosion.

    Regime 3 – Singularity avoidance:
        Feed x → 0.
        Penalty: output must NOT blow up (|f(x)| < 1e4).

    Regime 4 – Occam complexity (entropy):
        Penalise uniform distributions over operators — encourage crisp selection.
    """
    rng_c, rng_q, rng_s = jax.random.split(rng, 3)

    # Regime 1: Classical
    x_c = jnp.linspace(1.0, 5.0, 32)
    t_c = jnp.zeros(32)
    y_c, op_probs = model.apply(params, x_c, t_c, rng_c, tau=tau)
    target_c  = -x_c                          # dv/dt = -x  (harmonic)
    loss_c    = jnp.mean((y_c - target_c) ** 2)

    # Regime 2: Quantum (bounded output at micro scale)
    x_q = jnp.linspace(1e-3, 0.1, 32)
    t_q = jnp.zeros(32)
    y_q, _ = model.apply(params, x_q, t_q, rng_q, tau=tau)
    loss_q  = jnp.mean(y_q ** 2) * 0.1       # encourage near-zero micro output

    # Regime 3: Singularity avoidance
    x_s = jnp.array([1e-4, 1e-5, 1e-6])
    t_s = jnp.zeros(3)
    y_s, _ = model.apply(params, x_s, t_s, rng_s, tau=tau)
    loss_s  = jnp.mean(jnp.maximum(jnp.abs(y_s) - 1e4, 0.0)) * 1e-3

    # Regime 4: Occam / entropy (encourage confident discrete selection)
    # Stronger penalty = more pressure to collapse the superposition
    entropy  = -jnp.mean(jnp.sum(op_probs * jnp.log(op_probs + 1e-8), axis=-1))
    loss_ent = 0.10 * entropy   # 5× stronger than v1

    total = loss_c + loss_q + loss_s + loss_ent

    metrics = dict(total=total, classical=loss_c,
                   quantum=loss_q, singularity=loss_s, entropy=loss_ent)
    return total, metrics


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Crystallization — extract discrete equation from learned logits
# ─────────────────────────────────────────────────────────────────────────────

def crystallize(params: dict, model: LatentDreamEngine) -> str:
    """
    Returns a human-readable equation string from the argmax of learned logits.
    """
    logits   = params["params"]["node_logits"]     # [N, V]
    best_ops = [Op(int(jnp.argmax(logits[i]))) for i in range(logits.shape[0])]

    node_exprs = []
    for i, op in enumerate(best_ops):
        r = OP_REPR[op]
        if i == 0:
            left_s, right_s = "x", "t"
        elif i == 1:
            left_s, right_s = "t", "x"
        else:
            left_s  = node_exprs[i - 2]
            right_s = node_exprs[i - 1]

        # Format expression
        if op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV):
            expr = f"({left_s} {r} {right_s})"
        elif op in (Op.SIN, Op.COS, Op.EXP, Op.LOG, Op.NEG, Op.INV, Op.SQRT):
            expr = f"{r}({left_s})"
        elif op == Op.POW2:
            expr = f"({left_s})²"
        elif op in (Op.VAR_X, Op.VAR_T, Op.CONST_1, Op.CONST_PI):
            expr = r
        else:
            expr = f"{r}({left_s})"

        node_exprs.append(expr)

    return node_exprs[-1]


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(num_steps: int = 300, num_nodes: int = 10,
          tau_hi: float = 5.0, tau_lo: float = 0.05,
          lr: float = 0.05, seed: int = 42):

    print("=" * 70)
    print("  PROJECT BUDDHA: CONTINUOUS LATENT DREAM ENGINE")
    print("  Differentiable AST Discovery via Gumbel-Softmax STE")
    print("=" * 70)

    rng  = jax.random.PRNGKey(seed)
    model = LatentDreamEngine(num_nodes=num_nodes)

    # Init
    rng, init_rng = jax.random.split(rng)
    dummy_x = jnp.ones(4)
    params  = model.init(init_rng, dummy_x, dummy_x, init_rng, tau=1.0)

    n_params = sum(v.size for v in jax.tree.leaves(params))
    print(f"\n  Nodes: {num_nodes}  |  Vocab: {VOCAB}  |  Parameters: {n_params:,}")
    print(f"  Temperature: {tau_hi} → {tau_lo} over {num_steps} steps")
    print("-" * 70)

    opt   = optax.chain(
        optax.clip_by_global_norm(1.0),       # gradient clipping
        optax.adam(lr),
    )
    state = opt.init(params)

    @jax.jit
    def step(params, state, rng, tau):
        def loss_fn(p):
            return paradox_loss(p, model, rng, tau)
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return params, state, loss, metrics

    best_params = params
    best_loss   = float("inf")

    for s in range(1, num_steps + 1):
        tau  = anneal_temperature(s, num_steps, tau_hi, tau_lo)
        rng, step_rng = jax.random.split(rng)
        params, state, loss, metrics = step(params, state, step_rng, tau)

        loss_val = float(loss)
        if loss_val < best_loss and not jnp.isnan(loss):
            best_loss   = loss_val
            best_params = params

        if s % (num_steps // 10) == 0 or s == 1:
            print(f"  Step {s:4d}/{num_steps} | τ={tau:.3f} | "
                  f"Loss={loss_val:8.4f} | "
                  f"Classical={float(metrics['classical']):.4f} | "
                  f"Entropy={float(metrics['entropy']):.4f}")

    print("\n" + "=" * 70)
    print("  CRYSTALLIZATION — Superposition Collapses into Discrete Equation")
    print("=" * 70)

    # Show top-2 operators per node at the end
    final_logits = best_params["params"]["node_logits"]
    print("\n  Node-by-node operator confidences:")
    for i in range(num_nodes):
        probs = jax.nn.softmax(final_logits[i])
        top2  = jnp.argsort(-probs)[:2]
        p0, p1 = float(probs[top2[0]]), float(probs[top2[1]])
        n0, n1 = Op(int(top2[0])).name, Op(int(top2[1])).name
        bar = "█" * int(p0 * 30) + "░" * (30 - int(p0 * 30))
        print(f"  Node {i:2d}  [{bar}]  {n0} ({p0:.2%}) > {n1} ({p1:.2%})")

    eq = crystallize(best_params, model)
    print(f"\n  ✦ Discovered Equation (Root Output):")
    print(f"    f(x, t) = {eq}")
    print(f"\n  Best Paradox Loss: {best_loss:.6f}")
    print("=" * 70)

    return best_params, eq, best_loss


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(num_steps=500, num_nodes=10, tau_hi=5.0, tau_lo=0.02, lr=0.08)
