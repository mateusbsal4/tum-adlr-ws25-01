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
        self.current_gravity = -10.0
        self.current_wind_power = 0.0
        self.current_turbulence_power = 0.0
        
        # Call parent constructor with default values
        super().__init__(
            render_mode=render_mode,
            gravity=self.current_gravity,
            enable_wind=True,
            wind_power=self.current_wind_power,
            turbulence_power=self.current_turbulence_power
        )
        
        # Define disturbance ranges
        self.gravity_range = (-11.0, -9.0)  # Gravity variation around -10
        self.wind_power_range = (0.0, 15.0)  # Wind power from 0 to 15
        self.turbulence_power_range = (0.0, 1.5)  # Turbulence from 0 to 1.5
        
        # Store current disturbance configuration
        self.current_disturbance = {
            'gravity': self.current_gravity,
            'wind_power': self.current_wind_power,
            'turbulence_power': self.current_turbulence_power
        }
    
    def sample_disturbances(self):
        """Sample new random disturbance values within defined ranges"""
        self.current_gravity = np.random.uniform(*self.gravity_range)
        self.current_wind_power = np.random.uniform(*self.wind_power_range)
        self.current_turbulence_power = np.random.uniform(*self.turbulence_power_range)
        
        # Update current disturbance dictionary
        self.current_disturbance = {
            'gravity': self.current_gravity,
            'wind_power': self.current_wind_power,
            'turbulence_power': self.current_turbulence_power
        }
        
        # Update environment parameters
        self.gravity = self.current_gravity
        self.wind_power = self.current_wind_power
        self.turbulence_power = self.current_turbulence_power
    
    def set_disturbances(self, gravity=None, wind_power=None, turbulence_power=None):
        """Manually set specific disturbance values"""
        if gravity is not None:
            self.current_gravity = np.clip(gravity, *self.gravity_range)
        if wind_power is not None:
            self.current_wind_power = np.clip(wind_power, *self.wind_power_range)
        if turbulence_power is not None:
            self.current_turbulence_power = np.clip(turbulence_power, *self.turbulence_power_range)
        
        # Update current disturbance dictionary
        self.current_disturbance = {
            'gravity': self.current_gravity,
            'wind_power': self.current_wind_power,
            'turbulence_power': self.current_turbulence_power
        }
        
        # Update environment parameters
        self.gravity = self.current_gravity
        self.wind_power = self.current_wind_power
        self.turbulence_power = self.current_turbulence_power
    
    def get_current_disturbance(self):
        """Return current disturbance configuration"""
        return self.current_disturbance.copy()
    
    def reset(self, seed=None, options=None):
        """Reset environment and sample new disturbances"""
        # Sample new disturbances
        self.sample_disturbances()
        
        # Reset Box2D world with new gravity
        observation, info = super().reset(seed=seed, options=options)
        
        return observation, info
    
    def step(self, action):
        """Execute environment step with current disturbances"""
        state, reward, terminated, truncated, info = super().step(action)
        
        # Add disturbance information to info dict
        info['disturbance'] = self.get_current_disturbance()
        
        return state, reward, terminated, truncated, info

   