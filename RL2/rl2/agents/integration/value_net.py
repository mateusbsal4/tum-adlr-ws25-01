"""
Implements StatefulValueNet class.
"""

from typing import Union, Tuple, Optional, TypeVar, Generic
import logging

import torch as tc


ArchitectureState = TypeVar('ArchitectureState')


class StatefulValueNet(tc.nn.Module, Generic[ArchitectureState]):
    def __init__(self, preprocessing, architecture, value_head):
        super().__init__()
        self._preprocessing = preprocessing
        self._architecture = architecture
        self._value_head = value_head
        self._device = None

    def to(self, device):
        """Override to method to track device."""
        self._device = device
        return super().to(device)

    def cuda(self, device=None):
        """Override cuda method to track device."""
        self._device = tc.device('cuda', device)
        return super().cuda(device)

    def cpu(self):
        """Override cpu method to track device."""
        self._device = tc.device('cpu')
        return super().cpu()

    def _validate_inputs(
        self,
        curr_obs: Union[tc.LongTensor, tc.FloatTensor],
        prev_action: tc.LongTensor,
        prev_reward: tc.FloatTensor,
        prev_done: tc.FloatTensor
    ) -> None:
        """Validate input tensors."""
        if not all(isinstance(x, tc.Tensor) for x in [curr_obs, prev_action, prev_reward, prev_done]):
            raise TypeError("All inputs must be torch tensors")

        if tc.isnan(curr_obs).any():
            raise ValueError("NaN values detected in observations")
        if tc.isnan(prev_reward).any():
            raise ValueError("NaN values detected in rewards")

        if tc.isinf(curr_obs).any():
            raise ValueError("Infinite values detected in observations")
        if tc.isinf(prev_reward).any():
            raise ValueError("Infinite values detected in rewards")

        # Check batch dimensions match
        batch_size = curr_obs.size(0)
        if prev_action.size(0) != batch_size or prev_reward.size(0) != batch_size or prev_done.size(0) != batch_size:
            raise ValueError("Batch dimensions of inputs do not match")

    def _ensure_device(self, *tensors) -> Tuple[tc.Tensor, ...]:
        """Ensure all tensors are on the correct device."""
        if self._device is None:
            self._device = next(self.parameters()).device

        return tuple(x.to(self._device) if x is not None else None for x in tensors)

    def initial_state(self, batch_size: int) -> Optional[ArchitectureState]:
        """Initialize architecture state."""
        try:
            return self._architecture.initial_state(batch_size=batch_size)
        except Exception as e:
            logging.error(f"Error initializing value network state: {str(e)}")
            raise

    def forward(
        self,
        curr_obs: Union[tc.LongTensor, tc.FloatTensor],
        prev_action: tc.LongTensor,
        prev_reward: tc.FloatTensor,
        prev_done: tc.FloatTensor,
        prev_state: Optional[ArchitectureState]
    ) -> Tuple[tc.FloatTensor, ArchitectureState]:
        """
        Runs preprocessing and the architecture's state update;
        returns value estimate(s) and new state.

        Args:
            curr_obs: current observation(s) tensor with shape [B, ..., ?].
            prev_action: previous action(s) tensor with shape [B, ...]
            prev_reward: previous rewards(s) tensor with shape [B, ...]
            prev_done: previous done flag(s) tensor with shape [B, ...]
            prev_state: the architecture's previous state.

        Notes:
            '...' must be either one dimensional or must not exist

        Returns:
            Tuple containing value estimate(s) with batch shape [B, ...]
               and the architecture's new state.

        Raises:
            TypeError: If inputs are not torch tensors
            ValueError: If inputs contain invalid values or have mismatched dimensions
            RuntimeError: For other processing errors
        """
        try:
            # Move inputs to correct device
            curr_obs, prev_action, prev_reward, prev_done = self._ensure_device(
                curr_obs, prev_action, prev_reward, prev_done
            )
            if prev_state is not None:
                if isinstance(prev_state, tuple):
                    prev_state = tuple(s.to(self._device) if s is not None else None for s in prev_state)
                else:
                    prev_state = prev_state.to(self._device)

            # Validate inputs
            self._validate_inputs(curr_obs, prev_action, prev_reward, prev_done)

            # Preprocess inputs
            try:
                inputs = self._preprocessing(
                    curr_obs, prev_action, prev_reward, prev_done)
            except Exception as e:
                logging.error(f"Error in preprocessing step: {str(e)}")
                raise RuntimeError(f"Preprocessing failed: {str(e)}")

            # Run architecture
            try:
                features, new_state = self._architecture(
                    inputs=inputs, prev_state=prev_state)
            except Exception as e:
                logging.error(f"Error in architecture forward pass: {str(e)}")
                raise RuntimeError(f"Architecture forward pass failed: {str(e)}")

            # Generate value estimate
            try:
                value = self._value_head(features)
            except Exception as e:
                logging.error(f"Error in value head: {str(e)}")
                raise RuntimeError(f"Value head failed: {str(e)}")

            # Validate outputs
            if not isinstance(value, tc.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(value)}")
            if tc.isnan(value).any():
                raise ValueError("NaN values detected in value output")
            if tc.isinf(value).any():
                raise ValueError("Infinite values detected in value output")

            return value, new_state

        except Exception as e:
            logging.error(f"Error in value network forward pass: {str(e)}")
            raise
