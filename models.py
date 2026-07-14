import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    # emb_out_dim must include the dense feature width when dense is used
    def __init__(self, emb_module: nn.Module, emb_out_dim: int):
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
                 dropout: float = 0.1, use_bn: bool = True,
                 reg_weight: float = 1e-5):
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

    def _init_weights(self):
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
            # DCN-V2: normalized continuous features are part of x0
            e = torch.cat([e, dense], dim=1)
        return self.output(self.dnn(self.cross_network(e))).squeeze(1)

    def reg_loss(self) -> torch.Tensor:
        return self.reg_weight * sum(W.norm(2) for W in self.cross_w)
