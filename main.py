import argparse
import torch
from train import train
from game import Board
from mcts import MCTS
from net import TTTNet


parser = argparse.ArgumentParser(description="Alpha-0 tic tac toe")
parser.add_argument("--N", type=int, default=3)                                 # Board size
parser.add_argument("--W", type=int, default=3)                                 # Win con
parser.add_argument("--channels", type=int, default=64)                         # Net channels
parser.add_argument("--n_res", type=int, default=4)                             # Residual blocks
parser.add_argument("--sims_per_move", type=int, default=100)                   # MCTS sims
parser.add_argument("--episodes_per_iter", type=int, default=8)                 # Eps
parser.add_argument("--iters", type=int, default=50)                            # Iters
parser.add_argument("--batch_size", type=int, default=512)                      # Batch size
parser.add_argument("--replay_capacity", type=int, default=20000)               # Replay buffer
parser.add_argument("--lr", type=float, default=1e-3)                           
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
parser.add_argument("--print_every", type=int, default=10)
parser.add_argument("--logdir", type=str, default="./logdir")

args = parser.parse_args()

# Train
net = train(
    N=args.N,
    W=args.W,
    channels=args.channels,
    n_res=args.n_res,
    sims_per_move=args.sims_per_move,
    episodes_per_iter=args.episodes_per_iter,
    iters=args.iters,
    batch_size=args.batch_size,
    replay_capacity=args.replay_capacity,
    lr=args.lr,
    weight_decay=args.weight_decay,
    device_str=args.device,
    ckpt_dir=args.ckpt_dir,
    print_every=args.print_every,
    logdir=args.logdir,
)


