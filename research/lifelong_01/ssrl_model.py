"""
research/lifelong_01/ssrl_model.py -- Minimal fixed-capacity MLP for SSRL-01.

Architecture:
- Input dimension: 32 (16 features + 16 binary mask).
- Hidden Layer 1: Linear(32, 64) + ReLU
- Hidden Layer 2: Linear(64, 64) + ReLU
- Output: Linear(64, 2)
- Total Parameters: 6,402 (No BatchNorm, No LayerNorm, No Dropout).

Guarded with main check: imports do not construct or instantiate models.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class SSRLInstructionMLP(nn.Module):
    """
    Normalization-free multi-layer perceptron for instruction-conditioned parity selection.
    """

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64, output_dim: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, 32) concatenating features and selection mask.
        Returns:
            logits: Tensor of shape (batch_size, 2).
        """
        h1 = self.relu1(self.fc1(x))
        h2 = self.relu2(self.fc2(h1))
        logits = self.fc3(h2)
        return logits


def get_expected_parameter_count(input_dim: int = 32, hidden_dim: int = 64, output_dim: int = 2) -> int:
    """
    Computes exact parameter count mathematically without instantiating torch parameters.
    """
    p1 = input_dim * hidden_dim + hidden_dim
    p2 = hidden_dim * hidden_dim + hidden_dim
    p3 = hidden_dim * output_dim + output_dim
    return p1 + p2 + p3


if __name__ == "__main__":
    count = get_expected_parameter_count()
    print(f"Calculated Parameter Count: {count}")
    assert count == 6402
