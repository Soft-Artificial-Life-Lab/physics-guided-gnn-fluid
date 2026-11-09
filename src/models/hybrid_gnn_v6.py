# -*- coding: utf-8 -*-
"""v6 - dimension-parameterised form of the v4 Hybrid Physics-Guided GNN.

Version ID : v6_20261021_145750
Base       : src/models/hybrid_gnn_v4.py (HybridPhysicsGNNv4)

Why this file exists
--------------------
v4 and v5 are wired to three spatial components in exactly three places:

    input_node_dim = 3 * history + 1        node features
    input_edge_dim = 4                      [rel_pos (3), distance]
    output_dim     = 3                      predicted velocity delta

WaterRamps -- the second public benchmark -- is two-dimensional, so those
three numbers have to follow the dataset. The project's versioning rule is to
add a new version rather than edit a released one, so this subclass
re-parameterises those three pieces and inherits everything else. The
encoder-processor-decoder structure, the attention gate, the type weighting
and the normalisation are the v4 objects themselves, not re-implementations,
which is what lets a 2D result be reported as the same method rather than as
a variant of it.

Parameter counts differ between dimensions, as they must: the node encoder,
the edge encoder and the decoder each change width at their dataset-facing
end. Everything between them is identical. At dim=3 this class reproduces
HybridPhysicsGNNv4 exactly, which `__main__` below asserts.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .hybrid_gnn_v4 import HybridPhysicsGNNv4
except ImportError:                       # executed as a script
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from src.models.hybrid_gnn_v4 import HybridPhysicsGNNv4

V6_VERSION_ID = "v6_20261021_145750"


class HybridPhysicsGNNv6(HybridPhysicsGNNv4):
    """v4 with the spatial dimension as a constructor argument.

    Input (raw, unnormalised):
      node_features : (N, dim*history + 1)  [vel history oldest->newest, type]
      edge_index    : (2, E)                [row=src, col=dst]
      edge_attr     : (E, dim + 1)          [x_dst - x_src, distance]

    Output:
      normalised velocity delta prediction (N, dim).
    """

    def __init__(
        self,
        dim: int = 3,
        history: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 10,
        use_learned_attention: bool = True,
        use_type_weighting: bool = True,
        boundary_importance: float = 2.0,
    ):
        super().__init__(
            history=history,
            input_edge_dim=dim + 1,
            hidden_dim=hidden_dim,
            output_dim=dim,
            num_layers=num_layers,
            use_learned_attention=use_learned_attention,
            use_type_weighting=use_type_weighting,
            boundary_importance=boundary_importance,
        )
        self.dim = dim

        # v4 sized these for dim=3; resize for this dataset. Assigning to an
        # existing buffer name rebinds the buffer, so they stay in the
        # state_dict and keep moving with .to(device).
        node_dim = dim * history + 1
        if node_dim != self.input_node_dim:
            self.input_node_dim = node_dim
            self.node_mean = torch.zeros(node_dim)
            self.node_std = torch.ones(node_dim)
            self.node_encoder = nn.Sequential(
                nn.Linear(node_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def predict_next_velocity(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Physical v_{t+1} = v_t + denorm(dv_hat).

        v_t is the newest history slot. v4 slices it at a stride of 3; this
        slices at the dataset's dimension.
        """
        pred_norm = self.forward(node_features, edge_index, edge_attr)
        d = self.dim
        v_t = node_features[:, d * (self.history - 1): d * self.history]
        return v_t + self.denormalize_output(pred_norm)

    def get_config(self) -> dict:
        return {
            "version": V6_VERSION_ID,
            "dim": self.dim,
            "history": self.history,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "use_learned_attention": self.use_learned_attention,
            "use_type_weighting": self.use_type_weighting,
            "boundary_importance": self.boundary_importance,
        }


def make_node_features(vel_hist: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
    """(C, N, D) + (N,) -> (N, D*C + 1). History flattened oldest->newest.

    Dimension-generic replacement for v4's version, which reshapes to 3*C.
    """
    C, N, D = vel_hist.shape
    flat = vel_hist.permute(1, 0, 2).reshape(N, D * C)
    return torch.cat([flat, types.unsqueeze(-1)], dim=-1)


if __name__ == "__main__":
    import numpy as np

    for dim in (2, 3):
        torch.manual_seed(0)
        m = HybridPhysicsGNNv6(dim=dim)
        N, E = 1500, 20000
        nf = torch.randn(N, m.input_node_dim)
        nf[:, -1] = (torch.rand(N) > 0.85).float()
        ei = torch.randint(0, N, (2, E))
        ea = torch.randn(E, m.input_edge_dim)
        out = m(nf, ei, ea)
        vn = m.predict_next_velocity(nf, ei, ea)
        assert out.shape == (N, dim) and vn.shape == (N, dim)
        print("dim=%d  node_dim=%2d edge_dim=%d  out=%s  params=%d"
              % (dim, m.input_node_dim, m.input_edge_dim, tuple(out.shape),
                 m.count_parameters()))

    # At dim=3 the subclass must be the v4 model, parameter for parameter.
    torch.manual_seed(0)
    a = HybridPhysicsGNNv6(dim=3)
    torch.manual_seed(0)
    b = HybridPhysicsGNNv4()
    assert a.count_parameters() == b.count_parameters(), "dim=3 must match v4"
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb), "state_dict keys diverged from v4"
    assert all(torch.equal(sa[k], sb[k]) for k in sa), "weights diverged from v4"
    print("dim=3 reproduces HybridPhysicsGNNv4 exactly (%d params)"
          % a.count_parameters())

    # make_node_features must agree with v4's at dim=3.
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2] / "scripts" / "training"))
    import importlib
    v4m = importlib.import_module("v4_20260707_102544_train_v4")
    vh = torch.randn(3, 50, 3)
    ty = (torch.rand(50) > 0.8).float()
    assert torch.equal(make_node_features(vh, ty), v4m.make_node_features(vh, ty))
    print("make_node_features agrees with v4 at dim=3")
