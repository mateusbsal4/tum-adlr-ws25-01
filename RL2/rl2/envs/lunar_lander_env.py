"""
Implements the Lunar Lander environment(s) from Gymnasium.
"""

from typing import Tuple

import numpy as np

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import os
from .var_lander_gym import LunarLanderTargetPos, register 

from rl2.envs.abstract import MetaEpisodicEnv


class LunarLanderEnv(MetaEpisodicEnv):
    """

    """
    def __init__(self, max_episode_length = 10_000, target_x = 0.0, target_y = 0.0):

        register(
            id="LunarLanderTargetPos",  
            entry_point=LunarLanderTargetPos,
        )          #environment needs to be registered once
        self.target_x = target_x
        self.target_y = target_y
        self.max_ep_length = max_episode_length
        self.env = gym.make("LunarLanderTargetPos", render_mode="rgb_array", target_x=self.target_x, target_y = self.target_y)
        self.env = TimeLimit(self.env, self.max_ep_length)      #Wrapper which sets truncated flag to True when maximum episode length is exceeded

    def max_episode_len(self) -> int:
        """
        Return the maximum episode length.
        """
        return self.max_ep_length

    def new_env(self) -> None:
        """
        Sample a new LunarLander from the distribution over environments, by changing the target position.
        Returns:
            None
        """
<<<<<<< HEAD
        self.target_x = np.random.uniform(-2.5, 2.5)        #Bounds of the LunarLander env
=======
        self.target_x = np.random.uniform(-2.5, 2.5)
>>>>>>> 18d48a441c9a1a6c10882ac64ee93fc6ad5daf5d
        self.target_y = np.random.uniform(-2.5, 2.5)
        self.env = gym.make("LunarLanderTargetPos", render_mode="rgb_array", target_x=self.target_x, target_y = self.target_y)
        self.env = TimeLimit(self.env, max_episode_steps=self.max_ep_length)

    def new_env_fixed_target(self, target_x, target_y) -> None:
        """
        Return a new LunarLander from the distribution over environments, by changing the target position.
        Returns:
            None
        """
        self.target_x = target_x
        self.target_y = target_y
        self.env = gym.make("LunarLanderTargetPos", render_mode="rgb_array", target_x=self.target_x, target_y = self.target_y)

    def reset(self) -> np.ndarray[np.float32]:
        """
        Reset the environment.

        Returns:    
            initial state.
        """
        state, _ = self.env.reset()
        return state

    def step(self, action) -> Tuple[np.ndarray[np.float32], float, bool, dict]:
        """
        Take action in the LunarLander, and observe next state, reward, done.

        Args:
            action: action corresponding to an arm index.

        Returns:
            new_state, reward, done, info.
        """

        state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        if done:
            state = self.reset()

        return state, reward, done, {}