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


class SSRLInputGatedMLP(nn.Module):
    """
    Mask-conditioned multiplicative input-gating MLP for SSRL-14.
    Backbone layers (fc1, fc2, fc3) are instantiated first to match SSRLInstructionMLP bit-for-bit
    under identical torch.manual_seed, followed by a zero-initialized bias-free mask-to-feature gate.
    """

    def __init__(
        self,
        feature_dim: int = 16,
        mask_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 2,
        gate_gain: float = 4.0,
    ):
        super().__init__()
        if feature_dim <= 0 or mask_dim <= 0 or hidden_dim <= 0 or output_dim <= 0 or gate_gain <= 0.0:
            raise ValueError("All SSRLInputGatedMLP dimensions and gate_gain must be positive.")
        self.feature_dim = feature_dim
        self.mask_dim = mask_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gate_gain = float(gate_gain)

        self.fc1 = nn.Linear(feature_dim + mask_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.fc_in_gate = nn.Linear(mask_dim, feature_dim, bias=False)
        nn.init.zeros_(self.fc_in_gate.weight)

    def compute_input_gate(self, mask: torch.Tensor) -> torch.Tensor:
        """Computes smooth positive input-coordinate gain g_in = 2 * sigmoid(gate_gain * W_gate * mask)."""
        assert mask.shape[-1] == self.mask_dim, f"Expected mask_dim={self.mask_dim}, got {mask.shape[-1]}"
        return 2.0 * torch.sigmoid(self.gate_gain * self.fc_in_gate(mask))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass gating features [:feature_dim] via learned mask-conditioned gain before [x_gated; mask].
        """
        expected_dim = self.feature_dim + self.mask_dim
        if x.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected last dimension {expected_dim} (feature_dim={self.feature_dim} + mask_dim={self.mask_dim}), "
                f"got {x.shape[-1]}"
            )
        feat = x[:, : self.feature_dim]
        mask = x[:, self.feature_dim :]
        g_in = self.compute_input_gate(mask)
        x_gated = feat * g_in
        u_gated = torch.cat([x_gated, mask], dim=-1)
        return self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(u_gated)))))


def get_input_gated_parameter_count(
    feature_dim: int = 16,
    mask_dim: int = 16,
    hidden_dim: int = 64,
    output_dim: int = 2,
) -> int:
    """
    Computes exact parameter count of SSRLInputGatedMLP mathematically without instantiating tensors.
    """
    return (mask_dim * feature_dim) + get_expected_parameter_count(feature_dim + mask_dim, hidden_dim, output_dim)


if __name__ == "__main__":
    count = get_expected_parameter_count()
    print(f"Calculated Parameter Count: {count}")
    assert count == 6402
    cond_count = get_conditioned_parameter_count()
    print(f"Calculated Conditioned Parameter Count: {cond_count}")
    assert cond_count == 6722
    in_gated_count = get_input_gated_parameter_count()
    print(f"Calculated Input-Gated Parameter Count: {in_gated_count}")
    assert in_gated_count == 6658
