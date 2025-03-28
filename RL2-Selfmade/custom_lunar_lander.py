import numpy as np
from gymnasium.envs.box2d.lunar_lander import FPS
from var_lander_gym import LunarLanderTargetPos 
from gymnasium.envs.registration import register

import Box2D
from Box2D.b2 import (
    edgeShape,
    circleShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
    contactListener,
)

# Register the custom environment
register(
    id='CustomLunarLander-v2',
    entry_point='custom_lunar_lander:CustomLunarLander',
    max_episode_steps=1000,
)

class CustomLunarLander(LunarLanderTargetPos):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        # self.target_x = 0.0  # Default target position (center)
    
    def set_target_position(self, x_position):
        """Set new target landing position (-1 to 1 range)"""
        # Clamp the position to valid range
        self.target_x = np.clip(x_position, 0.1, 0.9)  # Slightly inside boundaries
        # Reset the environment with new target
        self.reset()
    
    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        
        return state, reward, terminated, truncated, info

   