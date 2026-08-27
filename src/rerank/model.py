"""
MODULE 5 — Reranker Model
==========================
Lightweight MLP that maps a feature vector to a scalar ranking score.

Architecture:
    Input dim  = FEATURE_DIM_2D (5) for 2D-only model
                 FEATURE_DIM_3D (10) for full 3D model
    Hidden     = [64, 32]
    Output     = scalar score (unbounded)
    Activation = ReLU (with BatchNorm for training stability)

Forward:
    score = model(x)  # higher → more likely correct

Factory methods:
    model_2d = RankerMLP.for_2d()  # 5 input features
    model_3d = RankerMLP.for_3d()  # 10 input features
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class RankerMLP(nn.Module if _TORCH_AVAILABLE else object):
    """
    MLP scoring function for learning-to-rank.

    Args:
        input_dim:    Dimensionality of the input feature vector (default: 9).
        hidden_dims:  Sizes of hidden layers (default: [64, 32]).
        dropout:      Dropout probability applied between hidden layers.
        use_batch_norm: Whether to apply BatchNorm after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 10,     # default = FEATURE_DIM_3D
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for RankerMLP.")

        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]  # default MLP; pass [] for a pure linear scorer

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # Build layers
        layers: List[nn.Module] = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim

        # Output layer: single scalar score (no activation)
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

        # Weight initialisation: Kaiming uniform for ReLU networks
        self._init_weights()


    # Weight initialisation


    def _init_weights(self) -> None:
        for i, module in enumerate(self.modules()):
            if isinstance(module, nn.Linear):
                # Use Kaiming for hidden layers (ReLU), Xavier for output layer (no activation)
                if module.out_features == 1:
                    nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


    # Forward pass


    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Args:
            x: Feature tensor of shape (batch_size, input_dim) or (input_dim,).

        Returns:
            score: Tensor of shape (batch_size, 1) or scalar.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, input_dim)

        # BatchNorm1d requires batch_size > 1 during training.
        # For single-sample inference, temporarily switch to eval mode.
        was_training = self.training
        if x.shape[0] == 1 and was_training:
            self.eval()
            out = self.net(x)
            self.train()
            return out

        return self.net(x)  # (batch_size, 1)

    def score(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Convenience: forward pass returning flat (batch_size,) scores.
        """
        return self.forward(x).squeeze(-1)


    # NumPy interface (inference without gradients)


    @torch.no_grad()
    def score_numpy(self, x: np.ndarray) -> np.ndarray:
        """
        Score an array of feature vectors without gradient tracking.

        Args:
            x: np.ndarray of shape (N, input_dim).

        Returns:
            np.ndarray of shape (N,) — ranking scores.
        """
        self.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32)
        scores = self.score(x_tensor)
        return scores.cpu().numpy()


    # Persistence helpers


    def save(self, path: str) -> None:
        """Save model weights to a .pt file."""
        torch.save(self.state_dict(), path)
        logger.info("RankerMLP saved to %s", path)

    @classmethod
    def load(
        cls,
        path: str,
        input_dim: int = 9,
        hidden_dims: Optional[List[int]] = None,
        **kwargs,
    ) -> "RankerMLP":
        """Load model weights from a .pt file."""
        model = cls(input_dim=input_dim, hidden_dims=hidden_dims, **kwargs)
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)
        logger.info("RankerMLP loaded from %s", path)
        return model

    def __repr__(self) -> str:
        return (
            f"RankerMLP(input_dim={self.input_dim}, "
            f"hidden_dims={self.hidden_dims}, "
            f"dropout={self.dropout})"
        )


    # Factory methods


    @classmethod
    def for_2d(
        cls,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        **kwargs,
    ) -> "RankerMLP":
        """Create a model for 2D-only features (input_dim=5)."""
        return cls(input_dim=5, hidden_dims=hidden_dims, dropout=dropout, **kwargs)

    @classmethod
    def for_3d(
        cls,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        **kwargs,
    ) -> "RankerMLP":
        """Create a model for full 3D features (input_dim=10)."""
        return cls(input_dim=10, hidden_dims=hidden_dims, dropout=dropout, **kwargs)
