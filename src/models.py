"""
Neural network architectures for behavioral keystroke authentication.
"""

from typing import Dict, List
import torch
import torch.nn as nn


class BiometricMLP(nn.Module):
    """
    Three-layer Multi-Layer Perceptron (MLP: D -> 16 -> 16 -> 1)
    for binary verification (legitimate user vs zero-effort impostor).
    Provides raw logits (for numerically stable BCEWithLogitsLoss),
    calibrated verification probabilities via sigmoid, and explicit
    layer-wise parameter blocks [(W_1, b_1), (W_2, b_2), (W_3, b_3)]
    for Gauss-Seidel Block Coordinate Descent.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def layer_blocks(self) -> List[List[nn.Parameter]]:
        """Returns the B = 3 topological layer parameter blocks [(W_l, b_l)]."""
        return [
            [self.fc1.weight, self.fc1.bias],
            [self.fc2.weight, self.fc2.bias],
            [self.fc3.weight, self.fc3.bias],
        ]

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Returns pre-activation logits for numerically stable loss computation."""
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.fc3(h2).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted legitimate-user probability in [0, 1]."""
        return torch.sigmoid(self.forward_logits(x))


def clone_model_with_init(input_dim: int, base_state_dict: Dict[str, torch.Tensor]) -> BiometricMLP:
    """Creates a new BiometricMLP instance initialized with identical weights."""
    model = BiometricMLP(input_dim)
    model.load_state_dict({k: v.detach().clone() for k, v in base_state_dict.items()})
    return model
