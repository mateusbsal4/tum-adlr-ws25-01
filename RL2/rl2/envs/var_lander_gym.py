import time
import math
from typing import TYPE_CHECKING, Optional

from render_browser import render_browser
import numpy as np

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle
from gymnasium.utils.step_api_compatibility import step_api_compatibility
from gymnasium.envs.registration import register
from stable_baselines3 import PPO


# ------------------------ SETUP -------------------------------------------------------

try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e


if TYPE_CHECKING:
    import pygame


FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14
SIDE_ENGINE_AWAY = 12
MAIN_ENGINE_Y_LOCATION = (
    4  # The Y location of the main engine on the body of the Lander.
)

VIEWPORT_W = 600
VIEWPORT_H = 400


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (
            self.env.lander == contact.fixtureA.body
            or self.env.lander == contact.fixtureB.body
        ):
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


# ------------------------ ENVIRONMENT -------------------------------------------------------

class LunarLanderTargetPos(gym.Env, EzPickle):
    r"""
    ## Description
    This environment is a classic rocket trajectory optimization problem.
    According to Pontryagin's maximum principle, it is optimal to fire the
    engine at full throttle or turn it off. This is the reason why this
    environment has discrete actions: engine on or off.

    There are two environment versions: discrete or continuous.
    The landing pad is always at coordinates (0,0). The coordinates are the
    first two numbers in the state vector.
    Landing outside of the landing pad is possible. Fuel is infinite, so an agent
    can learn to fly and then land on its first attempt.

    To see a heuristic landing, run:
    ```shell
    python gymnasium/envs/box2d/lunar_lander.py
    ```

    ## Action Space
    There are four discrete actions available:
    - 0: do nothing
    - 1: fire left orientation engine
    - 2: fire main engine
    - 3: fire right orientation engine

    ## Observation Space
    The state is an 8-dimensional vector: the coordinates of the lander in `x` & `y`, its linear
    velocities in `x` & `y`, its angle, its angular velocity, and two booleans
    that represent whether each leg is in contact with the ground or not.

    ## Rewards
    After every step a reward is granted. The total reward of an episode is the
    sum of the rewards for all the steps within that episode.

    For each step, the reward:
    - is increased/decreased the closer/further the lander is to the landing pad.
    - is increased/decreased the slower/faster the lander is moving.
    - is decreased the more the lander is tilted (angle not horizontal).
    - is increased by 10 points for each leg that is in contact with the ground.
    - is decreased by 0.03 points each frame a side engine is firing.
    - is decreased by 0.3 points each frame the main engine is firing.

    The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.

    An episode is considered a solution if it scores at least **200 points**.

    ## Starting State
    The lander starts at the top center of the viewport with a random initial
    force applied to its center of mass.

    ## Episode Termination
    The episode finishes if:
    1) the lander crashes (the lander body gets in contact with the moon);
    2) the lander gets outside of the viewport (`x` coordinate is greater than 1);
    3) the lander is not awake. From the [Box2D docs](https://box2d.org/documentation/md__d_1__git_hub_box2d_docs_dynamics.html#autotoc_md61),
        a body which is not awake is a body which doesn't move and doesn't
        collide with any other body:
    > When Box2D determines that a body (or group of bodies) has come to rest,
    > the body enters a sleep state which has very little CPU overhead. If a
    > body is awake and collides with a sleeping body, then the sleeping body
    > wakes up. Bodies will also wake up if a joint or contact attached to
    > them is destroyed.

    ## Arguments

    Lunar Lander has a large number of arguments

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
    ...                enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<LunarLander<LunarLander-v3>>>>>

    ```

     * `continuous` determines if discrete or continuous actions (corresponding to the throttle of the engines) will be used with the
     action space being `Discrete(4)` or `Box(-1, +1, (2,), dtype=np.float32)` respectively.
     For continuous actions, the first coordinate of an action determines the throttle of the main engine, while the second
     coordinate specifies the throttle of the lateral boosters. Given an action `np.array([main, lateral])`, the main
     engine will be turned off completely if `main < 0` and the throttle scales affinely from 50% to 100% for
     `0 <= main <= 1` (in particular, the main engine doesn't work  with less than 50% power).
     Similarly, if `-0.5 < lateral < 0.5`, the lateral boosters will not fire at all. If `lateral < -0.5`, the left
     booster will fire, and if `lateral > 0.5`, the right booster will fire. Again, the throttle scales affinely
     from 50% to 100% between -1 and -0.5 (and 0.5 and 1, respectively).

    * `gravity` dictates the gravitational constant, this is bounded to be within 0 and -12. Default is -10.0

    * `enable_wind` determines if there will be wind effects applied to the lander. The wind is generated using
     the function `tanh(sin(2 k (t+C)) + sin(pi k (t+C)))` where `k` is set to 0.01 and `C` is sampled randomly between -9999 and 9999.

    * `wind_power` dictates the maximum magnitude of linear wind applied to the craft. The recommended value for
     `wind_power` is between 0.0 and 20.0.

    * `turbulence_power` dictates the maximum magnitude of rotational wind applied to the craft.
     The recommended value for `turbulence_power` is between 0.0 and 2.0.

    ## Version History
    - v3:
        - Reset wind and turbulence offset (`C`) whenever the environment is reset to ensure statistical independence between consecutive episodes (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/954)).
        - Fix non-deterministic behaviour due to not fully destroying the world (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/728)).
        - Changed observation space for `x`, `y`  coordinates from $\pm 1.5$ to $\pm 2.5$, velocities from $\pm 5$ to $\pm 10$ and angles from $\pm \pi$ to $\pm 2\pi$ (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/752)).
    - v2: Count energy spent and in v0.24, added turbulence with wind power and turbulence_power parameters
    - v1: Legs contact with ground added in state vector; contact with ground give +10 reward points, and -10 if then lose contact; reward renormalized to 200; harder initial random push.
    - v0: Initial version

    ## Notes

    There are several unexpected bugs with the implementation of the environment.

    1. The position of the side thrusters on the body of the lander changes, depending on the orientation of the lander.
    This in turn results in an orientation dependent torque being applied to the lander.

    2. The units of the state are not consistent. I.e.
    * The angular velocity is in units of 0.4 radians per second. In order to convert to radians per second, the value needs to be multiplied by a factor of 2.5.

    For the default values of VIEWPORT_W, VIEWPORT_H, SCALE, and FPS, the scale factors equal:
    'x': 10, 'y': 6.666, 'vx': 5, 'vy': 7.5, 'angle': 1, 'angular velocity': 2.5

    After the correction has been made, the units of the state are as follows:
    'x': (units), 'y': (units), 'vx': (units/second), 'vy': (units/second), 'angle': (radians), 'angular velocity': (radians/second)

    <!-- ## References -->

    ## Credits
    Created by Oleg Klimov
    """

    # ---------------------------------------------------------------------------------------
    # Metadata for the environment, indicating the possible render modes and desired FPS.
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        continuous: bool = False,
        #! IMPORTANT DISTURBANCES - These parameters allow you to modify physical forces like gravity & wind.
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
        target_x: float = 0.0,
        target_y: float = 0.0
    ):
        # ---------------------------------------------------------------------------------------
        # Initialize EzPickle to allow this environment to be easily pickled/unpickled.
        # This is helpful in parallel/distributed training and for saving/loading environments.
        EzPickle.__init__(
            self,
            render_mode,
            continuous,
            gravity,
            enable_wind,
            wind_power,
            turbulence_power,
            target_x,
            target_y
        )

        # ---------------------------------------------------------------------------------------
        # Sanity check for gravity: must be between -12.0 and 0.0 for this environment.
        # If it's outside this range, assert will raise an error.
        assert (
            -12.0 < gravity and gravity < 0.0
        ), f"gravity (current value: {gravity}) must be between -12 and 0"
        self.gravity = gravity

        # ---------------------------------------------------------------------------------------
        # If wind_power is outside recommended range [0.0, 20.0], log a warning (but don't fail).
        if 0.0 > wind_power or wind_power > 20.0:
            gym.logger.warn(
                f"wind_power value is recommended to be between 0.0 and 20.0, (current value: {wind_power})"
            )
        self.wind_power = wind_power

        # ---------------------------------------------------------------------------------------
        # If turbulence_power is outside recommended range [0.0, 2.0], log a warning (but don't fail).
        if 0.0 > turbulence_power or turbulence_power > 2.0:
            gym.logger.warn(
                f"turbulence_power value is recommended to be between 0.0 and 2.0, (current value: {turbulence_power})"
            )
        self.turbulence_power = turbulence_power

        # ---------------------------------------------------------------------------------------
        # Target position for the lander to aim for (instead of always landing at (0,0)).
        # This is a custom extension to make the landing pad location flexible.
        self.target_x = target_x
        self.target_y = target_y

        # ---------------------------------------------------------------------------------------
        # Whether wind effects are enabled in the environment.
        self.enable_wind = enable_wind

        # ---------------------------------------------------------------------------------------
        # Pygame-related attributes for rendering.
        # - self.screen will hold the main pygame Surface.
        # - self.clock keeps track of time/frame rate.
        # - self.isopen tracks if the window is still open (or closed by the user).
        self.screen: pygame.Surface = None
        self.clock = None
        self.isopen = True

        # ---------------------------------------------------------------------------------------
        # Create a Box2D world with the specified gravity. This is where all physics will happen.
        self.world = Box2D.b2World(gravity=(0, gravity))

        # References to key Box2D bodies or objects in the environment:
        # - self.moon might represent the ground or "moon" surface.
        # - self.lander will be the Box2D body for the lunar lander (once created).
        self.moon = None
        self.lander: Optional[Box2D.b2Body] = None
        self.particles = []  # List to hold any particle effects (e.g., smoke from engines).

        # ---------------------------------------------------------------------------------------
        # Used to store the reward from the previous step for shaping or debugging.
        self.prev_reward = None

        # ---------------------------------------------------------------------------------------
        # Indicates if we use a continuous action space (with 2D throttle values)
        # or a discrete set of actions (0,1,2,3).
        self.continuous = continuous

        # ---------------------------------------------------------------------------------------
        # !Define lower and upper bounds for the observation space.
        # This ensures the agent's observations stay within a known range in each dimension.
        low = np.array(
            [
                -2.5,  # x coordinate
                -2.5,  # y coordinate
                -10.0, # x velocity
                -10.0, # y velocity
                -2 * math.pi,  # angle (radians)
                -10.0, # angular velocity
                -0.0,  # left leg contact (boolean, but stored as float 0 or 1)
                -0.0,  # right leg contact (boolean, but stored as float 0 or 1)
            ]
        ).astype(np.float32)

        high = np.array(
            [
                2.5,   # x coordinate
                2.5,   # y coordinate
                10.0,  # x velocity
                10.0,  # y velocity
                2 * math.pi,  # angle
                10.0,  # angular velocity
                1.0,   # left leg contact
                1.0,   # right leg contact
            ]
        ).astype(np.float32)

        # ---------------------------------------------------------------------------------------
        # Create the observation space as a Box in 8D: [x, y, vx, vy, angle, ang_vel, leg1_contact, leg2_contact].
        # Gym uses this space definition to validate and shape observations for the agent.
        self.observation_space = spaces.Box(low, high)

        # ---------------------------------------------------------------------------------------
        # Define the action space. If 'continuous' is True, we have a 2D Box space:
        #   [main engine throttle, lateral engine throttle].
        # Otherwise, we have 4 discrete actions: 0 (no-op), 1 (fire left engine),
        # 2 (fire main engine), 3 (fire right engine).
        if self.continuous:
            # For continuous control:
            # - main engine: -1..0 means off,  0..+1 scales from 50% to 100% power
            # - side engine: -1..-0.5 means left engine, +0.5..+1.0 means right engine,
            #                and -0.5..0.5 means off.
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # For discrete control: 4 possible integer actions.
            self.action_space = spaces.Discrete(4)

        # ---------------------------------------------------------------------------------------
        # Store the chosen rendering mode for later use in the 'render()' method.
        self.render_mode = render_mode

    def _destroy(self):
        if not self.moon:
            return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # ---------------------------------------------------------------------------------------
        # Reset the environment's random seed (for reproducibility) and call the parent reset method.
        super().reset(seed=seed)
        
        # Destroy existing Box2D bodies/fixtures from a previous episode to start fresh.
        self._destroy()

        # BUG Workaround:
        # There's a known issue (#728) in Gymnasium where the old world isn't fully cleaned up.
        # Re-creating a new Box2D world ensures no leftover entities or collision state remain.
        self.world = Box2D.b2World(gravity=(0, self.gravity))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref

        # Game state flags/variables
        self.game_over = False
        self.prev_shaping = None

        # ---------------------------------------------------------------------------------------
        # Get the effective width (W) and height (H) in "Box2D units" (scaled down from pixels).
        W = VIEWPORT_W / SCALE # 600/30 = 20 (Box2DUnits)
        H = VIEWPORT_H / SCALE # 400/30 = ~13 (Box2DUnits)

        # ---------------------------------------------------------------------------------------
        # Define the terrain. This environment uses a chunk-based approach for the terrain's shape.
        # CHUNKS = number of segments in the terrain mesh.
        CHUNKS = 11 
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)] # chunk x-positions (0, 2, 4, 8, ..., 20)

        # A pre-defined array of heights (relative), which we'll transform into actual coordinates.
        fixed_heights = [1, 1.5, 2, 2, 2.5, 2.0, 2.0, 2.0, 2.5, 1.0, 1.5]

        # Convert these relative heights into something that fits on our viewport.
        # We multiply by (H / 10) and then *add H/4 so the terrain sits partially up in the environment*.
        height = np.array(fixed_heights) * (H / 10) + H / 4

        # ---------------------------------------------------------------------------------------
        # helipad_y and helipad_x define where the lander should actually land:
        # The user can customize target_x, target_y, which modifies the final pad location.
        self.helipad_y = (H / 10) * self.target_y + H / 4
        helipad_x = (W / 5) * self.target_x + W / 2

        # ---------------------------------------------------------------------------------------
        # Find the chunk index closest to helipad_x so we can flatten the terrain around that chunk.
        target_chunk_idx = np.argmin(np.abs(np.array(chunk_x) - helipad_x))
        print(target_chunk_idx)
        #!FIX: TREAT EDGE CASES = TARGET CHUNK IS FIRST/LAST IN THE LIST
        if target_chunk_idx == CHUNKS-1:
            target_chunk_idx -= 2       # -2 because later we add 2 
        elif target_chunk_idx == 0:
            target_chunk_idx += 2
            

        # We'll define two corners of the helipad around that target chunk index (left & right edges).
        self.helipad_x1 = chunk_x[target_chunk_idx - 1]
        self.helipad_x2 = chunk_x[target_chunk_idx + 1]

        # Flatten the terrain around the helipad to create a "landing pad."
        # This sets multiple adjacent height points to the same helipad_y value.
        height[target_chunk_idx - 2] = self.helipad_y
        height[target_chunk_idx - 1] = self.helipad_y
        height[target_chunk_idx + 0] = self.helipad_y
        height[target_chunk_idx + 1] = self.helipad_y
        height[target_chunk_idx + 2] = self.helipad_y

        # ---------------------------------------------------------------------------------------
        # Smooth the terrain to avoid harsh transitions between chunks.
        # We do this by averaging each height with its neighbors, except at the edges.
        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1]) if 0 < i < CHUNKS - 1 else height[i]
            for i in range(CHUNKS)
        ]

        # ---------------------------------------------------------------------------------------
        # Create the "moon" as a static body (non-moving, infinite mass) with an edge from (0,0) to (W,0).
        # This acts as the base ground. We'll then build the sloped terrain on top of it.
        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )

        # We'll keep track of polygons ("sky_polys") to render the sky above the terrain.
        self.sky_polys = []

        # ---------------------------------------------------------------------------------------
        # Build the chunked terrain as connected edges. For each pair of points, we create an edge fixture.
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            
            # Each sky_poly covers the area above the terrain up to H, so we can fill it in with color.
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        # Optional coloring for the moon object (used if the renderer references color1, color2).
        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        # ---------------------------------------------------------------------------------------
        # Create the Lander body at the top of the viewport. (initial_x, initial_y)
        # LANDER_POLY is a polygon defining the lander's shape (from the base environment code).
        # We'll also set the fixture properties like density, friction, etc.
        initial_y = VIEWPORT_H / SCALE
        initial_x = VIEWPORT_W / SCALE / 2
        
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
                ),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # This fixture only collides with ground
                restitution=0.0, # Bounciness factor
            ),
        )

        # Color attributes for rendering the lander (if the renderer uses them).
        self.lander.color1 = (128, 102, 230)
        self.lander.color2 = (77, 77, 128)

        # ---------------------------------------------------------------------------------------
        # Apply a small random force to the center of the lander, so it doesn't start fully stable.
        self.lander.ApplyForceToCenter(
            (
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            ),
            True,
        )

        # ---------------------------------------------------------------------------------------
        # If wind is enabled, set up random indices for wind and torque,
        # which will be used in the step() function to apply random disturbances.
        if self.enable_wind:
            self.wind_idx = self.np_random.integers(-9999, 9999)
            self.torque_idx = self.np_random.integers(-9999, 9999)

        # ---------------------------------------------------------------------------------------
        # Create the lander's legs. Each leg is a separate dynamic body attached by a revolute joint.
        # We shift them left/right from the main lander body and give them a small angle offset.
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),  # slight tilt
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001,  # only collides with ground
                ),
            )
            # Track whether each leg has contacted the ground.
            leg.ground_contact = False
            
            # Coloring for the legs.
            leg.color1 = (128, 102, 230)
            leg.color2 = (77, 77, 128)

            # -----------------------------------------------------------------------------------
            # Create a revolute joint between the lander and the leg with motor/limit constraints.
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,  # slow rotation inward
            )
            # Adjust the angle limits so the legs can move within a certain range.
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5

            # Attach the joint to the world.
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        # ---------------------------------------------------------------------------------------
        # 'drawlist' is used later in rendering. It contains objects we want to draw (lander + legs).
        self.drawlist = [self.lander] + self.legs

        # The environment typically returns an observation and any extra info. 
        # Here, we call step(0) to get the initial observation.
        # return self.step(0)[0], {}
    
        if self.render_mode == "human":
            self.render()
        return self.step(np.array([0, 0]) if self.continuous else 0)[0], {}


    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3,
            ),
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all_particle):
        while self.particles and (all_particle or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def step(self, action):
        # ---------------------------------------------------------------------------------------
        # Ensure the environment has been reset and the lander exists.
        assert self.lander is not None, "Lander does not exist. Make sure to call reset() first."

        # ---------------------------------------------------------------------------------------
        # WIND AND TURBULENCE
        # Apply wind and torque (if enabled) only while the lander is airborne (i.e., legs not in contact).
        if self.enable_wind and not (self.legs[0].ground_contact or self.legs[1].ground_contact):
            # wind_mag is computed with a non-repeating sine-based function, scaled by self.wind_power
            wind_mag = (
                math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + math.sin(math.pi * 0.01 * self.wind_idx)
                )
                * self.wind_power
            )
            self.wind_idx += 1

            # Apply linear force (wind) to the center of the lander
            self.lander.ApplyForceToCenter(
                (wind_mag, 0.0),
                True,
            )

            # torque_mag is computed similarly, scaled by self.turbulence_power
            torque_mag = (
                math.tanh(
                    math.sin(0.02 * self.torque_idx)
                    + math.sin(math.pi * 0.01 * self.torque_idx)
                )
                * self.turbulence_power
            )
            self.torque_idx += 1

            # Apply torque (rotational force) to simulate turbulence
            self.lander.ApplyTorque(
                torque_mag,
                True,
            )

        # ---------------------------------------------------------------------------------------
        # ACTION PRE-PROCESSING
        # If using continuous controls, clip values to [-1, 1].
        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float64)
        else:
            # Otherwise, check the action is within the discrete action space: {0,1,2,3}.
            assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid "

        # ---------------------------------------------------------------------------------------
        # COMPUTE ENGINE IMPULSES
        # tip and side are unit vectors based on the lander's current angle in the Box2D world.
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])  # 90° rotation of the tip vector

        # dispersion adds a small random offset to the engine exhaust location
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        # m_power = main engine power, s_power = side engine power
        m_power = 0.0

        # ---------------------------------------------------------------------------------------
        # MAIN ENGINE ACTION
        # If continuous: action[0] controls main engine throttle; if discrete: action == 2 means "fire main engine".
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action == 2):
            if self.continuous:
                # Map [0..1] into [0.5..1.0] (i.e., cannot use <50% throttle in this environment).
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5
                assert 0.5 <= m_power <= 1.0
            else:
                # Discrete case => full power
                m_power = 1.0

            # Ox, Oy define the offset for where the main engine thrust is applied.
            ox = (
                tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                + side[0] * dispersion[1]
            )
            oy = (
                -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                - side[1] * dispersion[1]
            )
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)

            # Create a visual particle if we're rendering (it does not affect physics).
            if self.render_mode is not None:
                p = self._create_particle(
                    3.5,          # speed factor for the particle
                    impulse_pos[0],
                    impulse_pos[1],
                    m_power,
                )
                p.ApplyLinearImpulse(
                    (ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power),
                    impulse_pos,
                    True,
                )

            # Apply the main engine impulse to the lander in the opposite direction.
            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        # ---------------------------------------------------------------------------------------
        # SIDE ENGINES (orientation/left-right thrusters)
        s_power = 0.0
        # If continuous: action[1] magnitude > 0.5 => side thrusters. If discrete: actions 1 or 3 => left/right thrusters.
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1, 3]):
            if self.continuous:
                # direction is +1 if action[1] > 0, or -1 if action[1] < 0
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert 0.5 <= s_power <= 1.0
            else:
                # For discrete: 1 => left thruster, 3 => right thruster
                direction = action - 2  # gives -1 or +1
                s_power = 1.0

            # Compute offset for side engine thrust
            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )

            # The code uses '17' instead of the defined SIDE_ENGINE_HEIGHT in part of the offset,
            # which effectively changes how the thrust is applied relative to the lander’s angle.
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )

            # Create a visual particle if we're rendering.
            if self.render_mode is not None:
                p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
                p.ApplyLinearImpulse(
                    (ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power),
                    impulse_pos,
                    True,
                )

            # Apply the side engine impulse in the opposite direction.
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        # ---------------------------------------------------------------------------------------
        # ADVANCE PHYSICS
        # Step the Box2D physics world forward by one frame: 1/FPS seconds.
        # 6*30 and 2*30 are velocity and position iteration constants (common Box2D parameters).
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        # ---------------------------------------------------------------------------------------
        # STATE CALCULATION
        # Extract lander position/velocity for the agent's observation.
        pos = self.lander.position
        vel = self.lander.linearVelocity

        # Build an 8D state vector:
        #  0) horizontal coordinate (normalized)
        #  1) vertical coordinate (normalized)
        #  2) horizontal velocity (normalized)
        #  3) vertical velocity (normalized)
        #  4) angle
        #  5) angular velocity (scaled)
        #  6) left leg contact (0 or 1)
        #  7) right leg contact (0 or 1)
        state = [
            (pos.x - W*self.target_x) / (W / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (H/ 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        assert len(state) == 8

        # ---------------------------------------------------------------------------------------
        # REWARD CALCULATION
        # shaping is a potential-based reward that accounts for:
        #  1) distance from the target landing position (self.target_x, self.target_y)
        #  2) total velocity
        #  3) absolute angle
        #  4) leg contacts
        # The agent is encouraged to reduce distance, velocity, angle, and to keep both legs on ground.
        shaping = (
            # -100 * np.sqrt((state[0] - self.target_x)**2 + (state[1] - self.target_y)**2)
            # - 100 * np.sqrt(state[2]**2 + state[3]**2)
             -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )

        # The difference in shaping from the previous step is given as reward.
        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # Additional penalty for using fuel:
        reward -= m_power * 0.30  # main engine penalty
        reward -= s_power * 0.03  # side engines penalty

        # ---------------------------------------------------------------------------------------
        # EPISODE TERMINATION CONDITIONS
        terminated = False
        if self.game_over or pos.x >W or pos.x<0 or pos.y<0 or pos.y>H:
            terminated = True
            reward = -100

        # If the lander is "sleeping" (i.e., not awake), it means it has come to rest => success
        if not self.lander.awake:
            terminated = True
            reward = +100

        # NOTE: Gymnasium's TimeLimit wrapper handles max episode steps => 'False' for truncation here.
        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(state, dtype=np.float32), reward, terminated, False, {}


    def render(self):
        # ---------------------------------------------------------------------------------------
        # 1. CHECK IF RENDER MODE IS SET
        # If 'render_mode' is None, warn the user that no rendering will happen.
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        # ---------------------------------------------------------------------------------------
        # 2. IMPORT PYGAME LIBRARIES (OPTIONALLY RAISE AN ERROR IF NOT INSTALLED)
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[box2d]"`'
            ) from e

        # ---------------------------------------------------------------------------------------
        # 3. SET UP DISPLAY AND CLOCK (IF "human" MODE)
        # If no screen yet, initialize Pygame and set the display size to (VIEWPORT_W, VIEWPORT_H).
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        # Create a clock if we haven't yet, used to manage frames per second.
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # ---------------------------------------------------------------------------------------
        # 4. CREATE A NEW SURFACE TO DRAW ON
        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

        # Scale the surface if needed, though here it’s just a placeholder (SCALE might be 1).
        pygame.transform.scale(self.surf, (SCALE, SCALE))
        # Fill the entire surface with white (R=255, G=255, B=255).
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())

        # ---------------------------------------------------------------------------------------
        # 5. MANAGE PARTICLES
        # Each particle has a 'time to live' (ttl). We reduce ttl, then set its color based on ttl.
        for obj in self.particles:
            obj.ttl -= 0.15
            # color1 and color2 fade as ttl decreases, never going below some minimum value (0.2).
            obj.color1 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )
            obj.color2 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )
        self._clean_particles(False)  # Removes expired particles

        # ---------------------------------------------------------------------------------------
        # 6. DRAW A GRID (OPTIONAL VISUAL AID)
        # We iterate in steps of grid_spacing in both x and y to draw small squares.
        grid_spacing = 50
        grid_color = (200, 0, 0)  # Red squares
        for i in range(0, VIEWPORT_W, grid_spacing):
            for j in range(0, VIEWPORT_H, grid_spacing):
                pygame.draw.rect(
                    self.surf,
                    grid_color,
                    pygame.Rect(i - 5, j - 5, 10, 10)
                )

        # ---------------------------------------------------------------------------------------
        # 7. DRAW THE "SKY" POLYGONS
        # sky_polys is a list of quads going from the terrain's top to the top of the viewport.
        for p in self.sky_polys:
            scaled_poly = []
            for coord in p:
                scaled_poly.append((coord[0] * SCALE, coord[1] * SCALE))
            # Fill with black, then draw anti-aliased edges.
            pygame.draw.polygon(self.surf, (0, 0, 0), scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, (0, 0, 0))

        # ---------------------------------------------------------------------------------------
        # 8. DRAW PARTICLES & OTHER OBJECTS (LANDER, LEGS, ETC.)
        # self.drawlist typically holds the lander and legs; self.particles holds smoke/exhaust.
        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    # If the fixture is a circle, draw two overlapping circles (inner + outer).
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                else:
                    # If the fixture is a polygon, compute its path in pixels,
                    # draw a solid polygon, an anti-aliased polygon, and an outline.
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                    gfxdraw.aapolygon(self.surf, path, obj.color1)
                    pygame.draw.aalines(self.surf, color=obj.color2, points=path, closed=True)

                # ---------------------------------------------------------------------------------------
                # HELIPAD FLAGS
                # The helipad_x1 and helipad_x2 define the range of the pad. We draw a small flag on each side.
                for x in [self.helipad_x1, self.helipad_x2]:
                    x = x * SCALE
                    flagy1 = self.helipad_y * SCALE
                    flagy2 = flagy1 + 50
                    # Draw a vertical line
                    pygame.draw.line(
                        self.surf,
                        color=(255, 255, 255),
                        start_pos=(x, flagy1),
                        end_pos=(x, flagy2),
                        width=1,
                    )
                    # Draw a triangular flag
                    pygame.draw.polygon(
                        self.surf,
                        color=(204, 204, 0),
                        points=[
                            (x, flagy2),
                            (x, flagy2 - 10),
                            (x + 25, flagy2 - 5),
                        ],
                    )
                    gfxdraw.aapolygon(
                        self.surf,
                        [(x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5)],
                        (204, 204, 0),
                    )

        # ---------------------------------------------------------------------------------------
        # 9. FLIP SURFACE
        # Flip vertically to match the coordinate system used by Box2D.
        self.surf = pygame.transform.flip(self.surf, False, True)

        # ---------------------------------------------------------------------------------------
        # 10. DISPLAY OR RETURN RENDER
        if self.render_mode == "human":
            # If in "human" mode, draw the surface on the screen, handle events, control FPS, and update the display.
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()  # Process user events (like closing the window)
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            # If the user wants an array, convert the surface to a NumPy array and transpose it to [H,W,RGB].
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )


    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

