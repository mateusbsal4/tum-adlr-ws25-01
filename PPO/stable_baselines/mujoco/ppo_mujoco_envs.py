import os
os.environ['MUJOCO_GL'] = 'egl'

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
# from render_browser import render_browser

# @render_browser
def test_policy():
    vec_env = make_vec_env("Walker2d-v5", n_envs=1, env_kwargs={"render_mode": "human"})   #2d walker 
    # vec_env = make_vec_env("Ant-v5", n_envs=1, env_kwargs={"render_mode": "human"}) 
    model = PPO("MlpPolicy", vec_env, verbose=1, device = "cuda:0")
    model.learn(total_timesteps=25000)
    obs = vec_env.reset()
    while True:
        vec_env.render()
        action, _ = model.predict(obs)
        obs, _, _, _ = vec_env.step(action)
        # yield vec_env.render(mode="rgb_array")

test_policy()
