"""DNN architecture replicated from Notebook 05 so saved .pt checkpoints
can be reloaded in the serving process without importing the notebook.
Must stay in sync with the `DNN` class defined in
notebooks/05_deep_neural_network1.ipynb.
"""
from __future__ import annotations

import torch
from torch import nn


class DNN(nn.Module):
    """3-hidden-layer DNN with BatchNorm and Dropout (matches Notebook 05)."""

    def __init__(self, n_features: int, n_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256, momentum=0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128, momentum=0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64, momentum=0.01),
            nn.Dropout(dropout_rate),

            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
