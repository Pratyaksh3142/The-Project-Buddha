import math
import numpy as np

class MCTSNode:
    def __init__(self, state, parent=None, action_taken=None, prior_prob=0.0):
        self.state = state          # The Computation Graph DAG
        self.parent = parent
        self.action_taken = action_taken # The graph mutation that led here
        
        self.children = {}          # action -> MCTSNode
        
        self.visit_count = 0        # N(s, a)
        self.value_sum = 0.0        # W(s, a)
        self.q_value = 0.0          # Q(s, a) = W / N
        self.prior_prob = prior_prob # P(s, a) from the JAX Neural Network
        
        self.is_expanded = False

    def expand(self, action_probs):
        """
        Expands the node using the policy probabilities from the AlphaMath Brain.
        action_probs: dictionary mapping legal_action -> probability
        """
        for action, prob in action_probs.items():
            if action not in self.children:
                self.children[action] = MCTSNode(
                    state=self.state.copy_and_apply(action), # Hypothetical future graph
                    parent=self,
                    action_taken=action,
                    prior_prob=prob
                )
        self.is_expanded = True

    def select_child(self, c_puct=1.5):
        """
        Selects the child with the highest Upper Confidence Bound (PUCT).
        U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            # The exploration term encourages visiting nodes with high neural net 
            # prior probabilities (P) and low visit counts (N).
            exploration_term = c_puct * child.prior_prob * (
                math.sqrt(self.visit_count) / (1 + child.visit_count)
            )
            puct_score = child.q_value + exploration_term
            
            if puct_score > best_score:
                best_score = puct_score
                best_action = action
                best_child = child
                
        return best_action, best_child

    def backpropagate(self, value):
        """Passes the simulated physical value back up the tree."""
        self.visit_count += 1
        self.value_sum += value
        self.q_value = self.value_sum / self.visit_count
        if self.parent is not None:
            self.parent.backpropagate(value)


class GraphMCTS:
    def __init__(self, neural_network, jax_params, num_simulations=800):
        self.nn = neural_network
        self.params = jax_params
        self.num_simulations = num_simulations
        
        # JIT compile the forward pass for extreme speed
        import jax
        self._jitted_forward = jax.jit(self.nn.apply)

    def search(self, root_state):
        """
        Executes the Monte Carlo Tree Search to find the best mathematical move.
        """
        root = MCTSNode(state=root_state)

        for _ in range(self.num_simulations):
            node = root
            
            # 1. SELECTION: Traverse down the tree using PUCT until a leaf is found
            while node.is_expanded and node.children:
                _, node = node.select_child()
                
            # 2. EXPANSION & EVALUATION
            # Convert graph to JAX tensors
            node_features, adj_mask = node.state.to_jax_tensors()
            
            # Query the JAX AlphaMathBrain
            policy_logits, value = self._jitted_forward(
                self.params, node_features, adj_mask
            )
            
            # Mask out illegal topological moves (e.g. creating cycles in the DAG)
            legal_actions = node.state.get_legal_actions()
            action_probs = self._get_masked_probabilities(policy_logits, legal_actions)
            
            if not node.state.is_terminal():
                node.expand(action_probs)
                
            # 3. BACKPROPAGATION: Send the Neural Value prediction up the tree
            # value is shape (1, 1), we need a scalar
            val_scalar = float(value.squeeze())
            node.backpropagate(val_scalar)

        # 4. ACTION SELECTION: Return an improved policy based on visit counts
        action_visits = {
            action: child.visit_count 
            for action, child in root.children.items()
        }
        
        return self._compute_pi(action_visits)

    def _get_masked_probabilities(self, logits, legal_actions):
        # Softmax over only mathematically valid DAG mutations
        import jax.numpy as jnp
        import jax.nn as jnn
        probs = {}
        # ... logic to apply softmax strictly on legal_actions ...
        return probs

    def _compute_pi(self, action_visits, tau=1.0):
        # pi(a|s) = N(s,a)^(1/tau) / sum_b N(s,b)^(1/tau)
        total = sum(v ** (1/tau) for v in action_visits.values())
        return {a: (v ** (1/tau)) / total for a, v in action_visits.items()}
