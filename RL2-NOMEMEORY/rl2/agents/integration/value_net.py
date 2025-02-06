"""
Implements ValueNet class without memory component.
"""

from typing import Union, Tuple

import torch as tc


class StatefulValueNet(tc.nn.Module):
    def __init__(self, preprocessing, architecture, value_head):
        super().__init__()
        self._preprocessing = preprocessing
        self._architecture = architecture
        self._value_head = value_head

    def forward(
        self,
        curr_obs: Union[tc.LongTensor, tc.FloatTensor],
        prev_action: tc.LongTensor,
        prev_reward: tc.FloatTensor,
        prev_done: tc.FloatTensor
    ) -> tc.FloatTensor:
        """
        Runs preprocessing and the architecture;
        returns value estimate(s).

        Args:
            curr_obs: current observation(s) tensor with shape [B, ..., ?].
            prev_action: previous action(s) tensor with shape [B, ...]
            prev_reward: previous rewards(s) tensor with shape [B, ...]
            prev_done: previous done flag(s) tensor with shape [B, ...]

        Notes:
            '...' must be either one dimensional or must not exist

        Returns:
            Value estimate(s) with batch shape [B, ...]
        """
        inputs = self._preprocessing(
            curr_obs, prev_action, prev_reward, prev_done)

        features = self._architecture(inputs=inputs)

        vpred = self._value_head(features)

        return vpred
