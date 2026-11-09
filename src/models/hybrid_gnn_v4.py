"""
Hybrid Physics-Guided GNN v4 — full MeshGraphNet/GNS-recipe upgrade.

Version ID : v4_20260707_102544
Baseline   : src/models/hybrid_gnn.py (paper model) + stabilisations from
             src/models/hybrid_gnn_tuned.py (LayerNorm, gate bias init).
Neither baseline file is modified; this is a separate model used by the
v4 training driver (scripts/training/v4_20260707_102544_train_v4.py).

Upgrades over HybridPhysicsGNN (each independently switchable for ablation):

  [U1] Delta output. The model predicts the NORMALISED velocity change
       dv = (v_{t+1} - v_t), not the absolute next velocity. Integration
       v_{t+1} = v_t + denorm(dv_hat) happens outside (see
       `predict_next_velocity`). Zero-centred, unit-variance targets are the
       single biggest optimisation stabiliser in the GNS/MGN recipe.

  [U2] Input/target normalisation. Dataset-level mean/std for node features,
       edge features, and delta targets are stored as registered buffers so
       they travel inside the checkpoint. See `set_normalizer_stats`.

  [U3] Translation-invariant node features. Absolute positions are REMOVED
       from node input; geometry enters only through relative edge features
       (rel_pos, distance) — exactly like GNS/MGN. Node input is a velocity
       history of C steps plus the particle type:
           n_i = [v_i^{t-C+1}; ... ; v_i^{t}; c_i]  in R^{3C+1}
       Default C=3 -> input_node_dim = 10.

  [U4] Edge features gain an explicit distance channel:
           e_ij = [x_j - x_i ; ||x_j - x_i||]  in R^4.

  [U5] LayerNorm in every MLP (encoder / edge / node / decoder hidden) and
       attention-gate bias initialised to +2.0 (sigmoid ~ 0.88) so the gate
       starts near-identity — both taken from hybrid_gnn_tuned.py where they
       were validated as "Tier 1/2" stabilisations.

  [U6] Deeper processor: default num_layers = 10 (paper model used 5).

Kept from the paper model (the KIIS2026 contributions):
  - Learned attention gate alpha_ij in [0,1] per edge.
  - Physics-based type weighting w_ij (boundary_importance beta = 2.0).
  - Edge update  dh_ij = w_ij * alpha_ij * EdgeMLP([h_i; h_j; h_ij]).
  - Residual connections on both edge and node streams.
  - Aggregation via index_add_ at the destination node.

Training-time upgrades (noise injection, pushforward loss, LR schedule) live
in the v4 training driver, not here — the model stays a pure function.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

V4_VERSION_ID = "v4_20260707_102544"


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, layernorm: bool = True) -> nn.Sequential:
    """2-hidden-layer MLP: Linear-LN-ReLU-Linear-LN-ReLU-Linear."""
    layers = [nn.Linear(in_dim, hidden_dim)]
    if layernorm:
        layers.append(nn.LayerNorm(hidden_dim))
    layers += [nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)]
    if layernorm:
        layers.append(nn.LayerNorm(hidden_dim))
    layers += [nn.ReLU(), nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*layers)


class AttentionGateV4(nn.Module):
    """Learned edge-importance gate. Bias init +2.0 -> starts near identity [U5]."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64):
        super().__init__()
        input_dim = 2 * node_dim + edge_dim
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        with torch.no_grad():
            self.gate_net[-2].bias.fill_(2.0)

    def forward(self, src, dst, edge_attr):
        return self.gate_net(torch.cat([src, dst, edge_attr], dim=-1))


class EdgeModelV4(nn.Module):
    """Edge update with attention gate + boundary type weighting (paper contribution)."""

    def __init__(
        self,
        edge_dim: int,
        node_dim: int,
        hidden_dim: int = 64,
        use_learned_attention: bool = True,
        use_type_weighting: bool = True,
        boundary_importance: float = 2.0,
    ):
        super().__init__()
        self.use_learned_attention = use_learned_attention
        self.use_type_weighting = use_type_weighting
        self.boundary_importance = boundary_importance

        self.edge_net = _mlp(2 * node_dim + edge_dim, hidden_dim, edge_dim)
        if use_learned_attention:
            self.attention_gate = AttentionGateV4(node_dim, edge_dim, hidden_dim)

    def forward(self, src, dst, edge_attr, particle_types=None, edge_index=None):
        edge_update = self.edge_net(torch.cat([src, dst, edge_attr], dim=-1))

        if self.use_learned_attention:
            edge_update = self.attention_gate(src, dst, edge_attr) * edge_update

        if self.use_type_weighting and particle_types is not None and edge_index is not None:
            row, col = edge_index
            is_boundary = (particle_types[row] > 0.5) | (particle_types[col] > 0.5)
            type_w = torch.where(
                is_boundary,
                torch.tensor(self.boundary_importance, device=edge_update.device),
                torch.tensor(1.0, device=edge_update.device),
            ).unsqueeze(-1)
            edge_update = type_w * edge_update

        return edge_update


class NodeModelV4(nn.Module):
    """Node update: aggregate incoming edge messages at destination, then MLP."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.node_net = _mlp(node_dim + edge_dim, hidden_dim, node_dim)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg = torch.zeros(x.size(0), edge_attr.size(1), device=x.device, dtype=x.dtype)
        agg.index_add_(0, col, edge_attr)
        return self.node_net(torch.cat([x, agg], dim=-1))


class HybridBlockV4(nn.Module):
    """One message-passing block with residual edge & node streams."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        use_learned_attention: bool = True,
        use_type_weighting: bool = True,
        boundary_importance: float = 2.0,
    ):
        super().__init__()
        self.edge_model = EdgeModelV4(
            edge_dim, node_dim, hidden_dim,
            use_learned_attention, use_type_weighting, boundary_importance,
        )
        self.node_model = NodeModelV4(node_dim, edge_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr, particle_types=None):
        row, col = edge_index
        edge_attr = edge_attr + self.edge_model(
            x[row], x[col], edge_attr,
            particle_types=particle_types, edge_index=edge_index,
        )
        x = x + self.node_model(x, edge_index, edge_attr)
        return x, edge_attr


class HybridPhysicsGNNv4(nn.Module):
    """
    v4 Hybrid Physics-Guided GNN (Encode-Process-Decode).

    Input (raw, unnormalised — normalisation happens inside forward):
      node_features : (N, 3*history + 1)  [vel history oldest->newest, type]
      edge_index    : (2, E)              [row=src, col=dst]
      edge_attr     : (E, 4)              [x_dst - x_src, distance]

    Output:
      normalised velocity delta prediction (N, 3).
      Use `predict_next_velocity` to get physical v_{t+1}.
    """

    def __init__(
        self,
        history: int = 3,
        input_edge_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 3,
        num_layers: int = 10,
        use_learned_attention: bool = True,
        use_type_weighting: bool = True,
        boundary_importance: float = 2.0,
    ):
        super().__init__()
        self.history = history
        self.input_node_dim = 3 * history + 1
        self.input_edge_dim = input_edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_learned_attention = use_learned_attention
        self.use_type_weighting = use_type_weighting
        self.boundary_importance = boundary_importance

        # [U2] Normaliser buffers — identity until set_normalizer_stats() is
        # called; saved/restored automatically with the state_dict.
        self.register_buffer("node_mean", torch.zeros(self.input_node_dim))
        self.register_buffer("node_std", torch.ones(self.input_node_dim))
        self.register_buffer("edge_mean", torch.zeros(input_edge_dim))
        self.register_buffer("edge_std", torch.ones(input_edge_dim))
        self.register_buffer("target_mean", torch.zeros(output_dim))
        self.register_buffer("target_std", torch.ones(output_dim))

        # Encoder [U5: LayerNorm]
        self.node_encoder = nn.Sequential(
            nn.Linear(self.input_node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(input_edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Processor [U6: default 10 layers]
        self.processor_blocks = nn.ModuleList([
            HybridBlockV4(
                hidden_dim, hidden_dim, hidden_dim,
                use_learned_attention, use_type_weighting, boundary_importance,
            )
            for _ in range(num_layers)
        ])

        # Decoder (no LayerNorm on the final output layer)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def set_normalizer_stats(
        self,
        node_mean, node_std, edge_mean, edge_std, target_mean, target_std,
        eps: float = 1e-8,
    ):
        """Install dataset statistics [U2]. Accepts numpy arrays or tensors."""
        def _t(v, like):
            t = torch.as_tensor(v, dtype=like.dtype, device=like.device)
            return t.reshape(like.shape)

        self.node_mean.copy_(_t(node_mean, self.node_mean))
        self.node_std.copy_(_t(node_std, self.node_std).clamp_min(eps))
        self.edge_mean.copy_(_t(edge_mean, self.edge_mean))
        self.edge_std.copy_(_t(edge_std, self.edge_std).clamp_min(eps))
        self.target_mean.copy_(_t(target_mean, self.target_mean))
        self.target_std.copy_(_t(target_std, self.target_std).clamp_min(eps))

    def normalize_target(self, delta_v: torch.Tensor) -> torch.Tensor:
        return (delta_v - self.target_mean) / self.target_std

    def denormalize_output(self, pred_norm: torch.Tensor) -> torch.Tensor:
        return pred_norm * self.target_std + self.target_mean

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the NORMALISED delta-velocity prediction (N, 3) [U1]."""
        # Particle type from the raw (unnormalised) last column.
        particle_types = node_features[:, -1]

        x = (node_features - self.node_mean) / self.node_std
        e = (edge_attr - self.edge_mean) / self.edge_std

        x = self.node_encoder(x)
        e = self.edge_encoder(e)

        for block in self.processor_blocks:
            x, e = block(x, edge_index, e, particle_types)

        return self.decoder(x)

    def predict_next_velocity(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Physical v_{t+1} = v_t + denorm(dv_hat). v_t is the newest history
        slot, i.e. node_features[:, 3*(history-1) : 3*history]."""
        pred_norm = self.forward(node_features, edge_index, edge_attr)
        v_t = node_features[:, 3 * (self.history - 1): 3 * self.history]
        return v_t + self.denormalize_output(pred_norm)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_attention_weights(self, node_features, edge_index, edge_attr):
        """First-layer attention weights for analysis figures."""
        if not self.use_learned_attention:
            return None
        x = (node_features - self.node_mean) / self.node_std
        e = (edge_attr - self.edge_mean) / self.edge_std
        x = self.node_encoder(x)
        e = self.edge_encoder(e)
        row, col = edge_index
        gate = self.processor_blocks[0].edge_model.attention_gate
        return gate(x[row], x[col], e).detach().cpu().numpy()

    def get_config(self) -> dict:
        return {
            "version": V4_VERSION_ID,
            "history": self.history,
            "input_edge_dim": self.input_edge_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
            "use_learned_attention": self.use_learned_attention,
            "use_type_weighting": self.use_type_weighting,
            "boundary_importance": self.boundary_importance,
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    model = HybridPhysicsGNNv4()
    N, E = 900, 12000
    nf = torch.randn(N, model.input_node_dim)
    nf[:, -1] = (torch.rand(N) > 0.85).float()      # ~15% boundary
    ei = torch.randint(0, N, (2, E))
    ea = torch.randn(E, model.input_edge_dim)

    out = model(nf, ei, ea)
    v_next = model.predict_next_velocity(nf, ei, ea)
    print(f"version          : {V4_VERSION_ID}")
    print(f"input_node_dim   : {model.input_node_dim}")
    print(f"normalised delta : {tuple(out.shape)}")
    print(f"physical v_next  : {tuple(v_next.shape)}")
    print(f"parameters       : {model.count_parameters():,}")
