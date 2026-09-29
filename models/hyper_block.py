import time
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MessageAgg(nn.Module):
    def __init__(self, agg_method="mean"):
        super().__init__()
        self.agg_method = agg_method

    def forward(self, X, path):
        """
            X: [n_node, dim]
            path: col(source) -> row(target)
        """
        X = torch.matmul(path, X)
        if self.agg_method == "mean":
            X = X / path.sum(dim=2, keepdim=True).clamp_min(1)
            return X
        elif self.agg_method == "sum":
            pass
        return X


class HyPConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc = nn.Linear(c1, c2)
        self.v2e = MessageAgg(agg_method="mean")
        self.e2v = MessageAgg(agg_method="mean")


    def forward(self, x, H):
        x = self.fc(x)
        # v -> e
        E = self.v2e(x, H.transpose(1, 2).contiguous())
        # e -> v
        x = self.e2v(E, H)

        return x


class HyperComputeModule(nn.Module):
    def __init__(self, c1, c2, threshold):
        super().__init__()
        if c1 != c2:
            raise ValueError('Hypergraph residual requires equal input and output channels.')
        self.threshold = threshold
        self.hgconv = HyPConv(c1, c2)
        self.norm = nn.LayerNorm(c2)
        self.act = nn.SiLU()

    def forward(self, x, valid_mask=None):
        """[B, N, L, D]: build an independent object hypergraph for each frame."""
        b, n, length, d = x.shape
        if valid_mask is None:
            valid_mask = torch.ones((b, n), dtype=torch.bool, device=x.device)
        valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)
        x = x.masked_fill(~valid_mask[:, :, None, None], 0)
        x = x.permute(0, 2, 1, 3).reshape(b * length, n, d)
        valid = valid_mask[:, None, :].expand(b, length, n).reshape(b * length, n)
        hg = (torch.cdist(x, x) < self.threshold)
        hg = hg & valid[:, :, None] & valid[:, None, :]
        x = self.hgconv(x, hg.to(x.dtype)) + x
        # Normalize each node independently so padding cannot affect real objects.
        x = self.act(self.norm(x)).masked_fill(~valid[:, :, None], 0)
        return x.reshape(b, length, n, d).permute(0, 2, 1, 3)
