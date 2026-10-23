"""CoarseGNN: GraphSAGE node scorer over the coarse document graph (spec §2.3, §3.1).

    x -> Linear -> ReLU -> [typed SAGE layer + residual] x L -> Linear -> logit per node

``typed=True``: one SAGEConv per edge type (hierarchical / sequential / reference),
averaged, so the model can treat "my parent section" differently from "the clause I
cite". ``typed=False`` is the spec's single GraphSAGE over all edges (an ablation).
``use_edges=False`` turns message passing off (a per-node MLP; ablation).
``skip=True`` ("jumping knowledge"): the output reads the node's own representation
before message passing alongside the one after it, so averaging over neighbours cannot
wash out what distinguishes a paragraph from its siblings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Sequence

import torch
from torch import nn
from torch_geometric.nn import SAGEConv

from .features import EDGE_TYPES


@dataclass
class ModelConfig:
    in_dim: int
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.1
    typed: bool = True
    use_edges: bool = True
    use_query: bool = True
    skip: bool = False
    edge_types: Sequence[str] = EDGE_TYPES

    def to_dict(self) -> Dict:
        out = asdict(self)
        out["edge_types"] = list(self.edge_types)
        return out


class CoarseGNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        h = config.hidden
        self.input = nn.Linear(config.in_dim, h)
        types = list(config.edge_types) if config.typed else ["all"]
        self.convs = nn.ModuleList(
            nn.ModuleDict({t: SAGEConv(h, h, aggr="mean") for t in types}) for _ in range(config.layers))
        self.norms = nn.ModuleList(nn.LayerNorm(h) for _ in range(config.layers))
        self.dropout = nn.Dropout(config.dropout)
        self.out = nn.Linear(2 * h if config.skip else h, 1)

    def forward(self, x: torch.Tensor, edges: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.out(self.dropout(self.hidden(x, edges))).squeeze(-1)

    @property
    def hidden_dim(self) -> int:
        return self.out.in_features

    def hidden(self, x: torch.Tensor, edges: Dict[str, torch.Tensor]) -> torch.Tensor:
        """The representation the output layer reads (G4 attends over it)."""
        h = torch.relu(self.input(x))
        h0 = h
        if self.config.use_edges:
            if not self.config.typed:
                merged = torch.cat([edges[t] for t in self.config.edge_types], dim=1)
                edges = {"all": merged}
            for convs, norm in zip(self.convs, self.norms):
                messages = [conv(h, edges[t]) for t, conv in convs.items()]
                h = norm(h + self.dropout(torch.relu(torch.stack(messages).mean(0))))
        if self.config.skip:
            h = torch.cat([h0, h], dim=-1)
        return h
