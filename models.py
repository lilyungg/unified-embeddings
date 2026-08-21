import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from embeddings import UnifiedEmbedding


class SimpleMLP(nn.Module):
    def __init__(self, emb_module: nn.Module, emb_out_dim: int) -> None:
        super().__init__()
        self.emb = emb_module
        self.mlp = nn.Sequential(
            nn.Linear(emb_out_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),         nn.ReLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x: torch.Tensor, dense: torch.Tensor = None) -> torch.Tensor:
        e = self.emb(x)
        if dense is not None and dense.shape[1]:
            e = torch.cat([e, dense], dim=1)
        return self.mlp(e).squeeze(1)


class DCNV2(nn.Module):
    def __init__(self, emb_module: nn.Module, emb_out_dim: int,
                 num_cross: int = 1, dnn_dims: tuple = (192,),
                 dropout: float = 0.0, use_bn: bool = False,
                 reg_weight: float = 1e-5) -> None:
        super().__init__()
        D = emb_out_dim
        self.reg_weight = reg_weight
        self.emb = emb_module

        self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(D, D)) for _ in range(num_cross)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(D))    for _ in range(num_cross)])

        layers, in_d = [], D
        for out_d in dnn_dims:
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(in_d, out_d))
            if use_bn:
                layers.append(nn.BatchNorm1d(out_d))
            layers.append(nn.ReLU())
            in_d = out_d
        self.dnn    = nn.Sequential(*layers)
        self.output = nn.Linear(in_d, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.cross_w: nn.init.xavier_normal_(p)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def cross_network(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for W, b in zip(self.cross_w, self.cross_b):
            x = x0 * (x @ W.T + b) + x
        return x

    def forward(self, x: torch.Tensor, dense: torch.Tensor = None) -> torch.Tensor:
        e = self.emb(x)
        if dense is not None and dense.shape[1]:
            e = torch.cat([e, dense], dim=1)
        return self.output(self.dnn(self.cross_network(e))).squeeze(1)

    def reg_loss(self) -> torch.Tensor:
        return self.reg_weight * sum(W.norm(2) for W in self.cross_w)


class TwoTower(nn.Module):
    def __init__(self, n_rows, d, k, n_user_feats, worst_init=False, seed=42,
                 score='dot', temp=0.05, probes=1) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_rows, d // probes)
        nn.init.uniform_(self.emb.weight, -0.05, 0.05)
        self.user_proj = nn.Linear(n_user_feats * d, k, bias=False)
        self.item_proj = nn.Linear(d, k, bias=False)
        self.d, self.n_user_feats = d, n_user_feats
        self.score, self.temp = score, temp
        if worst_init:
            g = torch.Generator().manual_seed(seed)
            W0 = torch.randn(k, d, generator=g)
            W0 = W0 / W0.norm() * self.item_proj.weight.norm()
            with torch.no_grad():
                for t in range(n_user_feats):
                    self.user_proj.weight[:, t * d:(t + 1) * d] = W0
                self.item_proj.weight.copy_(W0)

    def blocks(self):
        W = self.user_proj.weight
        bs = [W[:, t * self.d:(t + 1) * self.d] for t in range(self.n_user_feats)]
        return bs + [self.item_proj.weight]

    def user(self, codes):
        e = self.emb(codes).reshape(len(codes), -1)
        u = self.user_proj(e)
        return F.normalize(u, dim=1) / self.temp if self.score == 'cosine' else u

    def items(self, codes):
        v = self.item_proj(self.emb(codes).reshape(len(codes), -1))
        return F.normalize(v, dim=1) if self.score == 'cosine' else v

    def proj_overlap(self):
        bs = self.blocks()
        with torch.no_grad():
            vals = [ (bs[i] @ bs[j].T).norm().item()
                     / (bs[i].norm().item() * bs[j].norm().item() + 1e-12)
                     for i in range(len(bs)) for j in range(i + 1, len(bs)) ]
        return (float(np.mean(vals)), float(np.max(vals))) if vals else (0.0, 0.0)

    def ranks(self):
        with torch.no_grad():
            rs = []
            for A in self.blocks():
                s = torch.linalg.svdvals(A.float().cpu())
                rs.append(((s ** 2).sum() / (s[0] ** 2 + 1e-12)).item())
        return rs


class TwoTowerFeat(nn.Module):

    def __init__(self, n_rows, d, k, n_q, n_i, probes=1) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_rows, d // probes)
        nn.init.uniform_(self.emb.weight, -0.05, 0.05)
        self.user_proj = nn.Linear(n_q * d, k, bias=False)
        self.item_proj = nn.Linear(n_i * d, k, bias=False)
        self.d, self.n_q, self.n_i = d, n_q, n_i

    def user(self, codes):
        return self.user_proj(self.emb(codes).reshape(len(codes), -1))

    def items(self, codes):
        return self.item_proj(self.emb(codes).reshape(len(codes), -1))

    def blocks(self):
        bs = [self.user_proj.weight[:, t * self.d:(t + 1) * self.d]
              for t in range(self.n_q)]
        bs += [self.item_proj.weight[:, t * self.d:(t + 1) * self.d]
               for t in range(self.n_i)]
        return bs

    def proj_overlap(self):
        bs = self.blocks()
        with torch.no_grad():
            vals = [(bs[i] @ bs[j].T).norm().item()
                    / (bs[i].norm().item() * bs[j].norm().item() + 1e-12)
                    for i in range(len(bs)) for j in range(i + 1, len(bs))]
        return (float(np.mean(vals)), float(np.max(vals))) if vals else (0.0, 0.0)

    def ranks(self):
        with torch.no_grad():
            rs = []
            for A in self.blocks():
                s = torch.linalg.svdvals(A.float().cpu())
                rs.append(((s ** 2).sum() / (s[0] ** 2 + 1e-12)).item())
        return rs


class SingleLayer(nn.Module):

    def __init__(self, emb_levels: int, n_features: int, emb_dim: int, seed: int = 42) -> None:
        super().__init__()
        self.emb = UnifiedEmbedding(emb_levels, emb_dim)
        self.n_features, self.emb_dim = n_features, emb_dim
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(emb_dim, generator=g)
        v = v / v.norm()
        self.theta = nn.Parameter(v.repeat(n_features, 1).clone())
        self.bias  = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, dense=None) -> torch.Tensor:
        e = self.emb(x).view(-1, self.n_features, self.emb_dim)
        return (e * self.theta).sum(dim=(1, 2)) + self.bias
