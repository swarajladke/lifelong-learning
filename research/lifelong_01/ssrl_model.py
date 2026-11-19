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


class SSRLConditionedMLP(nn.Module):
    """
    LayerNorm + learned mask-conditioned hidden gating MLP for SSRL-09.
    Splits 32-dim input into 16 feature bits and 16 mask bits without hardcoding feat * mask.
    """

    def __init__(
        self,
        feature_dim: int = 16,
        mask_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.mask_dim = mask_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1_feat = nn.Linear(feature_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc1_gate = nn.Linear(mask_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass splitting features [:feature_dim] and mask [feature_dim:].
        """
        expected_dim = self.feature_dim + self.mask_dim
        if x.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected last dimension {expected_dim} (feature_dim={self.feature_dim} + mask_dim={self.mask_dim}), "
                f"got {x.shape[-1]}"
            )
        feat = x[:, : self.feature_dim]
        mask = x[:, self.feature_dim :]
        z1 = self.ln1(self.fc1_feat(feat))
        g1 = torch.sigmoid(self.fc1_gate(mask))
        h1 = g1 * self.relu1(z1)
        z2 = self.ln2(self.fc2(h1))
        h2 = self.relu2(z2)
        return self.fc3(h2)


def get_conditioned_parameter_count(
    feature_dim: int = 16,
    mask_dim: int = 16,
    hidden_dim: int = 64,
    output_dim: int = 2,
) -> int:
    """
    Computes exact parameter count of SSRLConditionedMLP mathematically without instantiating tensors.
    """
    p_fc1_feat = feature_dim * hidden_dim + hidden_dim
    p_ln1 = 2 * hidden_dim
    p_fc1_gate = mask_dim * hidden_dim + hidden_dim
    p_fc2 = hidden_dim * hidden_dim + hidden_dim
    p_ln2 = 2 * hidden_dim
    p_fc3 = hidden_dim * output_dim + output_dim
    return p_fc1_feat + p_ln1 + p_fc1_gate + p_fc2 + p_ln2 + p_fc3


if __name__ == "__main__":
    count = get_expected_parameter_count()
    print(f"Calculated Parameter Count: {count}")
    assert count == 6402
    cond_count = get_conditioned_parameter_count()
    print(f"Calculated Conditioned Parameter Count: {cond_count}")
    assert cond_count == 6722
