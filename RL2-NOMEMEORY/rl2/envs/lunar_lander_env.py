"""
Implements the Lunar Lander environment(s) from Gymnasium.
"""

from typing import Tuple

import numpy as np

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import os
# from .var_lander_gym import VanillaLunarLander, register 
from .vanilla_lander_env import VanillaLunarLander
from gymnasium import register

from rl2.envs.abstract import MetaEpisodicEnv

VIEWPORT_W = 600
VIEWPORT_H = 400
SCALE = 30.0
W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE
class LunarLanderEnv(MetaEpisodicEnv):
    """

    """
    def __init__(self, max_episode_length = 10_000, target_x = 0.5, target_y = 0.25):

        register(
            id="VanillaLunarLander",  
            entry_point=VanillaLunarLander,
        )          #environment needs to be registered once
        self.target_x = target_x
        self.target_y = target_y
        self.max_ep_length = max_episode_length
        # self.render_mode = "rgb_array"
        self.render_mode = "human"
        
        self.env = gym.make("VanillaLunarLander", render_mode=self.render_mode)
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
       
        self.target_x = 0.5
        self.target_y = 0.5
        
        self.env = gym.make("VanillaLunarLander", render_mode=self.render_mode)
        self.env = TimeLimit(self.env, max_episode_steps=self.max_ep_length)

    def new_env_fixed_target(self, target_x, target_y) -> None:
        """
        Return a new LunarLander from the distribution over environments, by changing the target position.
        Returns:
            None
        """
        self.target_x = target_x
        self.target_y = target_y

        self.env = gym.make("VanillaLunarLander", render_mode=self.render_mode)

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
        # if done:
        #     state = self.reset()

        return state, reward, done, {}