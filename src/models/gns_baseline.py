"""Faithful PyTorch reimplementation of DeepMind's GNS baseline.

Reference: A. Sanchez-Gonzalez et al., "Learning to Simulate Complex Physics
with Graph Networks," ICML 2020 (and the open-sourced TF implementation in
deepmind-research/learning_to_simulate).

Architecture (paper defaults):
  - Encoder: node/edge MLPs, hidden 128, 2 hidden layers, LayerNorm output.
  - Processor: 10 GN blocks with residual connections on node AND edge latents.
  - Decoder: node MLP (no LayerNorm) -> normalised acceleration (3).

Input features (paper, sec. B.1):
  - node: C=5 previous velocities (normalised by dataset vel stats) and the
    distance to each of the 6 container faces, clipped at the connectivity
    radius R and normalised by R.
  - edge: relative displacement / R and its magnitude / R.

Target: acceleration a_t = v_{t+1} - v_t (dt = 1 in model units), normalised
by the dataset acceleration stats. Semi-implicit Euler update:
  v_{t+1} = v_t + a,  x_{t+1} = x_t + v_{t+1}.

Boundary handling is exactly GNS's: no wall particles, only the clipped
boundary-distance node features.
"""
from __future__ import annotations

import torch
import torch.nn as nn

GNS_VERSION = "gns_baseline_icml2020"


def _mlp(in_dim, hidden, out_dim, layers=2, layer_norm=True):
    seq = []
    d = in_dim
    for _ in range(layers):
        seq += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    seq.append(nn.Linear(d, out_dim))
    if layer_norm:
        seq.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*seq)


class GNSBaseline(nn.Module):
    def __init__(self, history: int = 5, hidden: int = 128,
                 num_blocks: int = 10, dim: int = 3):
        super().__init__()
        self.history = history
        self.hidden = hidden
        self.num_blocks = num_blocks
        self.dim = dim
        node_in = history * dim + 2 * dim          # velocities + wall distances
        edge_in = dim + 1                          # rel disp + |rel|

        # dataset stats (registered so they persist in checkpoints)
        for name, size in (("vel_mean", dim), ("vel_std", dim),
                           ("acc_mean", dim), ("acc_std", dim)):
            self.register_buffer(name, torch.zeros(size) if "mean" in name
                                 else torch.ones(size))

        self.node_encoder = _mlp(node_in, hidden, hidden)
        self.edge_encoder = _mlp(edge_in, hidden, hidden)
        self.edge_blocks = nn.ModuleList(
            _mlp(3 * hidden, hidden, hidden) for _ in range(num_blocks))
        self.node_blocks = nn.ModuleList(
            _mlp(2 * hidden, hidden, hidden) for _ in range(num_blocks))
        self.decoder = _mlp(hidden, hidden, dim, layer_norm=False)

    def set_stats(self, vel_mean, vel_std, acc_mean, acc_std):
        for name, val in (("vel_mean", vel_mean), ("vel_std", vel_std),
                          ("acc_mean", acc_mean), ("acc_std", acc_std)):
            getattr(self, name).copy_(torch.as_tensor(val, dtype=torch.float32))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def build_features(self, pos, vel_hist, bounds_lo, bounds_hi, radius):
        """pos (N,3); vel_hist (C,N,3) oldest->newest; returns node features."""
        C, N, D = vel_hist.shape
        v = (vel_hist - self.vel_mean) / self.vel_std          # (C,N,3)
        v = v.permute(1, 0, 2).reshape(N, C * D)
        d_lo = torch.clamp((pos - bounds_lo) / radius, -1.0, 1.0)
        d_hi = torch.clamp((bounds_hi - pos) / radius, -1.0, 1.0)
        return torch.cat([v, d_lo, d_hi], dim=-1)

    def forward(self, node_feat, edge_index, edge_attr):
        x = self.node_encoder(node_feat)
        e = self.edge_encoder(edge_attr)
        row, col = edge_index[0], edge_index[1]
        for eb, nb in zip(self.edge_blocks, self.node_blocks):
            e = e + eb(torch.cat([x[row], x[col], e], dim=-1))
            agg = torch.zeros_like(x)
            agg.index_add_(0, col, e)
            x = x + nb(torch.cat([x, agg], dim=-1))
        return self.decoder(x)                                  # normalised acc

    def predict_next_velocity(self, pos, vel_hist, edge_index, edge_attr,
                              bounds_lo, bounds_hi, radius):
        nf = self.build_features(pos, vel_hist, bounds_lo, bounds_hi, radius)
        acc_n = self.forward(nf, edge_index, edge_attr)
        acc = acc_n * self.acc_std + self.acc_mean
        return vel_hist[-1] + acc
