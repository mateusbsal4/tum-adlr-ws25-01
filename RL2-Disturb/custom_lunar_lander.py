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
        super().__init__(
            render_mode=render_mode,
            enable_wind=True  # Enable wind in parent class
        )
        self.target_x = 0.5  # Default target position
        self.wind_power = 0.0  # Default wind power
        
    def reset(self, seed=None, options=None):
        """Reset the environment with current target position."""
        # Update target position in parent class
        
        
        # Reset environment
        return super().reset(seed=seed, options=options)
    
    def get_wind_power(self):
        return self.wind_power

    
    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        
        return state, reward, terminated, truncated, info

   