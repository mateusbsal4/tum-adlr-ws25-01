"""
Value heads for RL^2 agents.
"""

import torch as tc


class LinearValueHead(tc.nn.Module):
    """
    Value head for a reinforcement learning agent.
    """
    def __init__(self, num_features):
        super().__init__()
        self.net = tc.nn.Sequential(
            tc.nn.Linear(num_features, num_features // 2),
            tc.nn.ReLU(),
            tc.nn.Linear(num_features // 2, 1)
        )

    def forward(self, features: tc.FloatTensor) -> tc.FloatTensor:
        """
        Computes a value estimate from features and returns it.

        Args:
            features: tc.FloatTensor of features with shape [B, ..., F].

        Returns:
            tc.FloatTensor of value estimates with shape [B, ...].
        """
        # print("\nDEBUG - Value Head:")
        # print(f"Input features range: {features.min().item():.3f} to {features.max().item():.3f}")
        
        values = self.net(features).squeeze(-1)
        # print(f"Output values range: {values.min().item():.3f} to {values.max().item():.3f}")
        
        return values
