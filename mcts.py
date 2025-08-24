import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import numpy as np
import torch

@dataclass
class Node:
    state: object
    parent: Optional["Node"]
    prior: float                            # P(s,a) from parent
    action_from_parent: Optional[int]

    N_sa: np.ndarray              # Visit counts
    V_sa: np.ndarray              # Value sum
    Q_sa: np.ndarray              # Average value
    P_sa: np.ndarray              # Priors (post legal mask)

    policy_mask: np.ndarray       # Boolean legal mask
    children: Dict[int, "Node"]
    expanded: bool

    @staticmethod
    def create_root(state):
        tiles = state.N * state.N
        mask = state.policy_mask().cpu().numpy().astype(bool)
        zeros = np.zeros(tiles, dtype=np.float32)

        return Node(
            state=state, 
            parent=None, 
            prior=1.0, 
            action_from_parent=None,
            N_sa=zeros.copy(), 
            V_sa=zeros.copy(), 
            Q_sa=zeros.copy(), 
            P_sa=zeros.copy(),
            policy_mask=mask, 
            children={}, 
            expanded=False
        )

    def is_terminal(self):
        """
        Check if final state
        """
        done, value = self.state.outcome()
        return bool(done), float(value)

    def expand(self, priors):
        """
        Expand node
        Apply legal move mask to priors, normalise over remaining
        """
        prior = np.asarray(priors, dtype=np.float32)
        prior[~self.policy_mask] = 0.0
        sum = float(prior.sum())

        if sum > 0:
            prior /= sum

        else:
            legal = np.flatnonzero(self.policy_mask)
            if legal.size > 0:
                prior[legal] = 1.0 / legal.size

        self.P_sa = prior
        self.expanded = True

    def select_action(self, c_puct: float):
        """
        PUCT: argmax_a Q + U, where U = c * P * sqrt(sum N) / (1+N).
        Return idx of move with highest PUCT 
        """
        legal = self.policy_mask
        N_sum = float(self.N_sa.sum()) + 1e-8
        U = np.zeros_like(self.P_sa, dtype=np.float32)
        U[legal] = c_puct * self.P_sa[legal] * (math.sqrt(N_sum) / (1.0 + self.N_sa[legal]))
        score = self.Q_sa + U
        score[~legal] = -1e9
        return int(np.argmax(score))

    def child(self, a):

        if a in self.children:
            return self.children[a]
        
        next_state = self.state.play(a)
        A = next_state.N * next_state.N
        mask = next_state.policy_mask().cpu().numpy().astype(bool)
        zeros = np.zeros(A, dtype=np.float32)

        node = Node(
            state=next_state, parent=self, prior=float(self.P_sa[a]), action_from_parent=a,
            N_sa=zeros.copy(), V_sa=zeros.copy(), Q_sa=zeros.copy(), P_sa=zeros.copy(),
            policy_mask=mask, children={}, expanded=False
        )

        self.children[a] = node
        return node

    def backup(self, a, v):
        """
        Update edge stats for action a using value v using current player pov
        """
        self.N_sa[a] += 1.0
        self.V_sa[a] += v
        self.Q_sa[a] = self.V_sa[a] / self.N_sa[a]


class NetworkEval:
    def __init__(self, net, device):
        self.net = net
        self.device = device if device is not None else torch.device("cpu")

    @torch.no_grad()
    def evaluate(self, state):
        x = state.encode().unsqueeze(0).to(self.device)          # (1,2,N,N)
        logits, value = self.net(x)                               # (1,A), (1,1)
        priors = logits.squeeze(0).detach().cpu().numpy().astype(np.float32)
        v = float(value.squeeze().detach().cpu().item())
        return priors, v


class MCTS:
    def __init__(
        self,
        net: torch.nn.Module,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        device: Optional[torch.device] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.eval = NetworkEval(net, device=device)
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_eps = float(dirichlet_eps)
        self.rng = rng if rng is not None else np.random.default_rng()

    def run(self, root_state, n_simulations, noise=True):
        """
        Run simulations, return value/visit policy over a.
        """
        root = Node.create_root(root_state)

        # 0 policy for leaf nodes
        done, value = root.is_terminal()
        if done:
            A = root_state.N * root_state.N
            return np.zeros(A, dtype=np.float32)

        # Expand root
        priors, _ = self.eval.evaluate(root_state)
        root.expand(priors)

        # Dirichlet noise (exploration)
        if noise:
            legal = np.flatnonzero(root.policy_mask)

            if legal.size > 0:
                noise = self.rng.gamma(shape=self.dirichlet_alpha, scale=1.0, size=legal.size).astype(np.float32)
                noise = noise / (noise.sum() + 1e-8)
                p = root.P_sa.copy()
                p[legal] = (1.0 - self.dirichlet_eps) * p[legal] + self.dirichlet_eps * noise
                root.P_sa = p

        # Run sim
        for _ in range(int(n_simulations)):
            node = root
            path: List[Tuple[Node, int]] = []

            # Selection
            while True:
                done, terminal_val = node.is_terminal()
                if done:
                    leaf_value = terminal_val
                    break

                if not node.expanded:
                    # Expansion + evaluation
                    priors_leaf, value_leaf = self.eval.evaluate(node.state)
                    node.expand(priors_leaf)
                    leaf_value = value_leaf
                    break

                a = node.select_action(self.c_puct)
                path.append((node, a))
                node = node.child(a)

            # Backup (flip sign each ply)
            v = float(leaf_value)
            for parent, action in reversed(path):
                parent.backup(action, v)
                v = -v

        # Visit-count policy at root
        visits = root.N_sa.copy()
        visits[~root.policy_mask] = 0.0
        s = visits.sum()
        if s > 0:
            visits /= s
        return visits

    def select_move(self, root_state, num_simulations: int, temperature: float = 1.0) -> int:
        """
        Sample an action from visit counts using temp
        """
        pi = self.run(root_state, num_simulations, inject_noise_at_root=True)
        legal = root_state.policy_mask().cpu().numpy().astype(bool)
        p = pi.copy()
        p[~legal] = 0.0
        if temperature <= 1e-6:
            p[~legal] = -1.0
            return int(np.argmax(p))
        if temperature != 1.0:
            with np.errstate(divide='ignore', invalid='ignore'):
                p = np.power(p, 1.0 / float(temperature))
        s = p.sum()
        if s <= 0:
            return int(np.random.choice(np.flatnonzero(legal)))
        p /= s
        return int(np.random.choice(np.arange(p.size), p=p))
