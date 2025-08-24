import os
import math
import random
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from game import Board
from mcts import MCTS
from net import TTTNet

torch.set_float32_matmul_precision('high')

# Replay buffer
class Replay:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data = []  # (state_tensor, pi, z)

    def push(self, state, pi, z):
        # state: (2,N,N) tensor 
        # pi: action policy
        # z: -1, 0, 1
        self.data.append((state.detach().cpu(), pi.astype(np.float32), int(z)))

        if len(self.data) > self.capacity:
            self.data.pop(0)

    def __len__(self):
        return len(self.data)

    def sample(self, batch_size: int):
        batch = random.sample(self.data, batch_size)
        xs = torch.stack([b[0] for b in batch], dim=0)                       # (B,2,N,N)
        pis = torch.tensor(np.stack([b[1] for b in batch], axis=0))           # (B,A)
        zs = torch.tensor([b[2] for b in batch], dtype=torch.float32)         # (B,)
        return xs, pis, zs


@torch.no_grad()
def single_ep(N, W, mcts, simulations, temperature):
    """
    1 game of network guide mcts
    Returns state_tensors, policy_targets, value_targets
    """
    board = Board.new_board(N=N, W=W)
    states: List[torch.Tensor] = []
    policies: List[np.ndarray] = []

    played = 0
    done, value = board.outcome()

    while not done:
        pi = mcts.run(board, n_simulations=simulations, noise=True)  # (A,)
        print(pi)
        states.append(board.encode())                               # (2,N,N) float32 torch tensor (CPU)
        policies.append(pi)

        # Sample a move
        legal = board.policy_mask().cpu().numpy().astype(bool)
        p = pi.copy()
        p[~legal] = 0.0
        if temperature != 1.0:
            with np.errstate(divide='ignore', invalid='ignore'):
                p = np.power(p, 1.0 / max(1e-6, float(temperature)))
        ssum = p.sum()
        if ssum <= 0:
            a = int(np.random.choice(np.flatnonzero(legal)))
        else:
            p /= ssum
            a = int(np.random.choice(np.arange(p.size), p=p))

        board = board.play(a)
        done, _ = board.outcome()
        played += 1
        if played > N * N + 2:                  # inf loop
            break

    # Terminal value
    _, z_term = board.outcome()  # in {-1,0,1}
    T = len(states)
    values: List[int] = []

    for t in range(T):
        flips = (T - t) & 1
        z_t = z_term if flips == 0 else -z_term
        values.append(int(z_t))

    return states, policies, values, played, int(z_term)


# Loss (policy + value)
def combined_loss(pi_logits, v_pred, pi_target, z_target):
    """
    pi_logits: (B,A), v_pred: (B,)
    pi_target: (B,A) visit count, z_target: (B,) in [-1,0,1]
    """
    logp = torch.log_softmax(pi_logits, dim=1)
    policy_loss = -(pi_target * logp).sum(dim=1).mean()
    value_loss = torch.mean((z_target - v_pred) ** 2)
    return policy_loss + value_loss, policy_loss, value_loss

def entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # p assumed non-negative, sums to 1 along dim=1
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=1)

def kl_div(p_target: torch.Tensor, logp_pred: torch.Tensor) -> torch.Tensor:
    # KL(p || q) with q represented by logp_pred
    p = p_target.clamp_min(1e-8)
    return (p * (p.log() - logp_pred)).sum(dim=1)


# Training loop
def train(
    N: int = 3,
    W: int = 3,
    channels: int = 64,
    n_res: int = 4,
    sims_per_move: int = 200,
    episodes_per_iter: int = 25,
    iters: int = 200,
    batch_size: int = 128,
    replay_capacity: int = 20000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device_str: str = "cuda",
    ckpt_dir: str = "./checkpoints",
    print_every: int = 10,
    logdir: Optional[str] = "./logdir",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    os.makedirs(ckpt_dir, exist_ok=True)
    if logdir is not None:
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(log_dir=logdir)
    else:
        writer = None

    # Model
    net = TTTNet(N=N, fan_in=2, channels=channels, n_res=n_res).to(device)
    net = torch.compile(net)
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    replay = Replay(replay_capacity)

    # MCTS guided by current network
    mcts = MCTS(net=net, c_puct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.25, device=device)

    global_step = 0
    for it in range(1, iters + 1):
        # fill replay buffer with self play
        game_lengths = []
        term_vals = []  # terminal values from terminal state's POV
        for _ in range(episodes_per_iter):
            states, policies, zs, ply, z_term = single_ep(N, W, mcts, sims_per_move, temperature=1.0)
            game_lengths.append(ply)
            term_vals.append(z_term)
            for s_t, pi_t, z_t in zip(states, policies, zs):
                replay.push(s_t, pi_t, z_t)

        # Log stats
        if writer is not None:
            writer.add_scalar("selfplay/replay_size", len(replay), it)
            writer.add_scalar("selfplay/avg_game_len", float(np.mean(game_lengths)), it)
            writer.add_scalar("selfplay/draw_rate", float(np.mean([1 if v == 0 else 0 for v in term_vals])), it)
            writer.add_scalar("selfplay/win_rate_curPOV_terminal", float(np.mean([1 if v > 0 else 0 for v in term_vals])), it)

        if len(replay) < batch_size:
            continue  # not enough data yet

        # Updates
        net.train()
        running = {"loss": 0.0, "pl": 0.0, "vl": 0.0, "ent": 0.0, "kl": 0.0, "v_abs": 0.0}
        steps = 0

        for _ in range(200):
            x, pi_target, z_target = replay.sample(batch_size)
            x = x.to(device)                             # (B,2,N,N)
            pi_target = pi_target.to(device)             # (B,A)
            z_target = z_target.to(device)               # (B,)

            pi_logits, v_pred = net(x)                   # (B,A), (B,1) or (B,)
            v_pred = v_pred.view(-1)

            loss, pl, vl = combined_loss(pi_logits, v_pred, pi_target, z_target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

            with torch.no_grad():
                logp_pred = torch.log_softmax(pi_logits, dim=1)
                pi_pred = torch.softmax(pi_logits, dim=1)
                ent = entropy(pi_target).mean()          # entropy of target distribution
                kld = kl_div(pi_target, logp_pred).mean()
                v_abs = v_pred.abs().mean()

            running["loss"] += float(loss.item())
            running["pl"] += float(pl.item())
            running["vl"] += float(vl.item())
            running["ent"] += float(ent.item())
            running["kl"] += float(kld.item())
            running["v_abs"] += float(v_abs.item())
            steps += 1
            global_step += 1

            if writer is not None and (global_step % 10 == 0):
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/policy_loss", pl.item(), global_step)
                writer.add_scalar("train/value_loss", vl.item(), global_step)
                writer.add_scalar("train/target_entropy", ent.item(), global_step)
                writer.add_scalar("train/kl_pi_target_pred", kld.item(), global_step)
                writer.add_scalar("train/value_abs", v_abs.item(), global_step)

        if steps > 0 and (it % print_every == 0):
            avg = {k: v / max(1, steps) for k, v in running.items()}
            print(f"[iter {it}] loss={avg['loss']:.4f} policy={avg['pl']:.4f} value={avg['vl']:.4f} "
                  f"ent(target)={avg['ent']:.3f} KL(pi||pred)={avg['kl']:.3f} | "
                  f"v_abs={avg['v_abs']:.3f} replay={len(replay)}")

        if it % 50 == 0:
            path = os.path.join(ckpt_dir, f"ttt_net_iter_{it}.pt")
            torch.save({"iter": it, "model": net.state_dict(), "opt": opt.state_dict()}, path)

    # Final
    path = os.path.join(ckpt_dir, f"ttt_net_final.pt")
    torch.save({"iter": iters, "model": net.state_dict(), "opt": opt.state_dict()}, path)

    if writer is not None:
        writer.flush()
        writer.close()

    return net
