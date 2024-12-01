import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from render_browser import render_browser
import time

@render_browser
def test_policy():
    # Create and set up the environment using make_vec_env to wrap Gym
    # This will ensure we follow the VecEnv API (with only the observation returned)
    vec_env = make_vec_env("LunarLander-v3", n_envs=1, env_kwargs={"render_mode": "rgb_array"})

    # Load or train the model
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=25000)   
    model.save("ppo_lunar")

    # Reload the trained model
    model = PPO.load("ppo_lunar")

    # Reset the environment and run the policy
    obs = vec_env.reset()  # For SB3, this returns only `obs`, not a tuple.
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)

        # Yield the rendered frame as a NumPy array
        yield vec_env.render(mode="rgb_array")
        time.sleep(0.02)

# Run the policy and serve it in a browser
test_policy()

