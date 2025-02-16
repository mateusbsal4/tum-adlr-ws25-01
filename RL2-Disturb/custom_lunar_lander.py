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
        # Initialize with default disturbances
        self.current_wind_power = 0.0
        
        # Call parent constructor with default values
        super().__init__(
            render_mode=render_mode,
            gravity=-10.0,  # Fixed gravity
            enable_wind=True,
            wind_power=self.current_wind_power,
            turbulence_power=0.0  # No turbulence
        )
        
        # Define wind power range
        self.wind_power_range = (0.0, 15.0)  # Wind power from 0 to 15
        
        # Store current disturbance configuration
        self.current_disturbance = {
            'wind_power': self.current_wind_power
        }
    
    def sample_disturbances(self):
        """Sample new random wind power value within defined range"""
        self.current_wind_power = np.random.uniform(*self.wind_power_range)
        
        # Update current disturbance dictionary
        self.current_disturbance = {
            'wind_power': self.current_wind_power
        }
        
        # Update environment parameters
        self.wind_power = self.current_wind_power
    
    def set_disturbances(self, wind_power=None):
        """Manually set specific wind power value"""
        if wind_power is not None:
            self.current_wind_power = np.clip(wind_power, *self.wind_power_range)
        
        # Update current disturbance dictionary
        self.current_disturbance = {
            'wind_power': self.current_wind_power
        }
        
        # Update environment parameters
        self.wind_power = self.current_wind_power
    
    def get_current_disturbance(self):
        """Return current disturbance configuration"""
        return self.current_disturbance.copy()
    
    def reset(self, seed=None, options=None):
        """Reset environment and sample new wind power"""
        # Sample new wind power
        self.sample_disturbances()
        
        # Reset Box2D world
        observation, info = super().reset(seed=seed, options=options)
        
        return observation, info
    
    def step(self, action):
        """Execute environment step with current wind power"""
        state, reward, terminated, truncated, info = super().step(action)
        
        # Add disturbance information to info dict
        info['disturbance'] = self.get_current_disturbance()
        
        return state, reward, terminated, truncated, info

   