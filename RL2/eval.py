"""
Implements evaluation routine for a trained RL2 agent on an e
environment 
stateful meta-reinforcement learning agents.
"""

import torch as tc
import numpy as np

from rl2.envs.abstract import MetaEpisodicEnv
from rl2.agents.integration.policy_net import StatefulPolicyNet
#from render_browser import render_browser

#@render_browser
@tc.no_grad()
def evaluation_loop(
        env: MetaEpisodicEnv,
        policy_net: StatefulPolicyNet,
        target_x: float,    #must be within [-2.5, 2.5]!
        target_y: float     #must be within [-2.5, 2.5]!
    ) -> None:
    """
    Evaluates a trained RL2 agent on a new Lunar Lander environment with predefined landing position

    Args:
        env: environment.
        policy_net: policy network.
        target_x: x coordinate of the target landing position
        target_y: y coordinate of the target landing position

    Returns:
        None
    """

    env.new_env_fixed_target(target_x, target_y)        #create 

    o_t = np.array([env.reset()])
    a_tm1 = np.array([0])
    r_tm1 = np.array([0.0])
    d_tm1 = np.array([1.0])
    h_tm1_policy_net = policy_net.initial_state(batch_size=1)
    done_t = 0
    while not done_t:
        pi_dist_t, h_t_policy_net = policy_net(
            curr_obs=tc.FloatTensor(o_t),
            prev_action=tc.LongTensor(a_tm1),
            prev_reward=tc.FloatTensor(r_tm1),
            prev_done=tc.FloatTensor(d_tm1),
            prev_state=h_tm1_policy_net)

        a_t = pi_dist_t.sample()
        env.render()
        #yield env.render()
        o_tp1, r_t, done_t, _ = env.step(a_t.squeeze(0).detach().numpy().item())
        o_t = np.array([o_tp1])
        a_tm1 = np.array([a_t.squeeze(0).detach().numpy()])
        r_tm1 = np.array([r_t])
        d_tm1 = np.array([float(done_t)])
        h_tm1_policy_net = h_t_policy_net