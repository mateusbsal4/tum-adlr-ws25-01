import numpy as np
from gymnasium.envs.box2d.lunar_lander import FPS
from var_lander_gym import LunarLanderTargetPos 

import Box2D
from Box2D.b2 import (
    edgeShape,
    circleShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
    contactListener,
)

class CustomLunarLander(LunarLanderTargetPos):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.target_x = 0.0  # Default target position (center)
    
    def set_target_position(self, x_position):
        """Set new target landing position (-1 to 1 range)"""
        # Clamp the position to valid range
        self.target_x = np.clip(x_position, -0.8, 0.8)  # Slightly inside boundaries
        # Reset the environment with new target
        self.reset()
    
    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        
        # Add distance penalty to encourage moving toward target
        if not terminated and not truncated:
            x_pos = state[0]  # Current x position
            distance_to_target = abs(x_pos - self.target_x)
            reward -= 0.1 * distance_to_target
        
        return state, reward, terminated, truncated, info
    
    def _update_landing_pad(self):
        """Update landing pad position"""
        # Convert target_x from [-1, 1] to actual world coordinates
        world_target_x = self.target_x * (self.VIEWPORT_W/self.SCALE/2)
        
        # Destroy old landing pad and terrain
        if hasattr(self, 'landing_pad') and self.landing_pad in self.world.bodies:
            self.world.DestroyBody(self.landing_pad)
        
        # Create new landing pad at target position
        self.landing_pad = self.world.CreateStaticBody(
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(0, 0), (20/self.SCALE, 0),
                                           (20/self.SCALE, 5/self.SCALE),
                                           (0, 5/self.SCALE)]),
                friction=0.5,
                restitution=0,
            )
        )
        
        # Set position of new landing pad
        self.landing_pad.position = (world_target_x - 10/self.SCALE, 0.1)
        self.landing_pad_x = self.target_x  # Store current pad position
        
        # Update landing pad fixture category
        for fixture in self.landing_pad.fixtures:
            fixture.sensor = True 