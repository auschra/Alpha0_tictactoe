# board.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import torch

Plane = Tuple[int, ...]  # flat binary plane, length N*N, entries 0/1

def win_lines(N, W):
    """
    Return list of 1D index arrays. If line of length W contain all ones,
    that is a winning line
    """

    lines: List[np.ndarray] = []

    # Rows
    for r in range(N):
        for c0 in range(0, N - W + 1):
            idxs = [r * N + (c0 + dc) for dc in range(W)]
            lines.append(np.asarray(idxs, dtype=np.int32))

    # Columns
    for c in range(N):
        for r0 in range(0, N - W + 1):
            idxs = [(r0 + dr) * N + c for dr in range(W)]
            lines.append(np.asarray(idxs, dtype=np.int32))

    # Diagonal right
    for r0 in range(0, N - W + 1):
        for c0 in range(0, N - W + 1):
            idxs = [(r0 + k) * N + (c0 + k) for k in range(W)]
            lines.append(np.asarray(idxs, dtype=np.int32))

    # Diagonal left
    for r0 in range(W - 1, N):
        for c0 in range(0, N - W + 1):
            idxs = [(r0 - k) * N + (c0 + k) for k in range(W)]
            lines.append(np.asarray(idxs, dtype=np.int32))

    return lines

@dataclass(frozen=True)
class Board:
    """
    2 plane board in perspective of current player
    channel 0 (current) is plane to move
    Planes are flat, binary
    """
    N: int
    W: int
    current: Plane
    opponent: Plane
    _win_lines: Tuple[np.ndarray, ...] = ()

    @staticmethod
    def new_board(N= 3, W=3):
        zeros = (0,) * (N * N)
        return Board(N=N, W=W, current=zeros, opponent=zeros, _win_lines=tuple(win_lines(N, W)))
    
    def played_mask(self):
        """
        Boolean mask of occupied tiles
        """
        c = np.fromiter(self.current, dtype=np.uint8)
        o = np.fromiter(self.opponent, dtype=np.uint8)
        return (c | o).astype(bool)

    def legal_moves(self) -> np.ndarray:
        """
        Flat idx of empty cells
        """
        occupied = self.played_mask()
        return np.flatnonzero(~occupied)

    def _has_win(self, plane):
        """
        Check if given plane has won
        """
        p = np.fromiter(plane, dtype=np.uint8)

        for line in self._win_lines:
            if p[line].all():
                return True
        return False

    def finished(self) -> bool:
        """
        Check if game has terminated, either with a win or draw
        """
        return self._has_win(self.current) or self._has_win(self.opponent) or self.played_mask().all()

    def play(self, idx):
        """
        Apply move at idx for current player
        Return next state with flipped perspected
        """
        if idx < 0 or idx >= self.N * self.N:
            raise IndexError("move index out of bounds")
        if self.current[idx] == 1 or self.opponent[idx] == 1:
            raise ValueError("illegal move")

        current_list = list(self.current)
        current_list[idx] = 1
        current_after = tuple(current_list)

        return Board(N=self.N, W=self.W, current=self.opponent, opponent=current_after, _win_lines=self._win_lines)

    def outcome(self):
        """
        Returns (terminal, result), result is next board state, current player pov
        +1 = current player wins, -1 = current player loses, 0 = draw or not finished
        """
        if self._has_win(self.current):
            return True, +1
        if self._has_win(self.opponent):
            return True, -1
        if self.played_mask().all():
            return True, 0
        return False, 0

    def encode(self, dtype=np.float32):
        """
        Return (2, N, N) tensor,channel 0 = current, 1 = opponent
        """
        N = self.N
        c = np.fromiter(self.current, dtype=dtype).reshape(N, N)
        o = np.fromiter(self.opponent, dtype=dtype).reshape(N, N)
        x = np.stack([c, o], axis=0)

        return torch.from_numpy(x)

    def policy_mask(self):
        """
        Return (N*N,) bool tensor, True = available actions
        For inf softmax
        """
        m = self.legal_moves()
        mask = np.zeros(self.N * self.N, dtype=bool)
        mask[m] = True

        return torch.from_numpy(mask)

def encode_batch(states: List[Board], dtype=np.float32):

    if not states:
        raise ValueError("empty batch")
    N = states[0].N
    xs = [s.encode(dtype=dtype) for s in states]

    return torch.stack(xs, dim=0).contiguous()

def masks_batch(states: List[Board]) -> torch.Tensor:
    if not states:
        raise ValueError("empty batch")
    ms = [s.policy_mask() for s in states]
    return torch.stack(ms, dim=0)
