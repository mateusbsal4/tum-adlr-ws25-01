"""
Implements evaluation routine for a trained RL2 agent on an e
environment 
stateful meta-reinforcement learning agents.
"""

import torch as tc
import numpy as np

from rl2.envs.abstract import MetaEpisodicEnv
from rl2.agents.integration.policy_net import StatefulPolicyNet

import argparse
from functools import partial
import logging
import os

import torch as tc

from train import create_env, create_argparser, create_net, create_preprocessing

from rl2.envs.bandit_env import BanditEnv
from rl2.envs.mdp_env import MDPEnv
from rl2.envs.lunar_lander_env import LunarLanderEnv

from rl2.agents.preprocessing.lunar import LunarLanderPreprocessing
from rl2.agents.preprocessing.tabular import MABPreprocessing, MDPPreprocessing
from rl2.agents.architectures.gru import GRU
from rl2.agents.architectures.lstm import LSTM
from rl2.agents.architectures.snail import SNAIL
from rl2.agents.architectures.transformer import Transformer
from rl2.agents.heads.policy_heads import LinearPolicyHead
from rl2.agents.heads.value_heads import LinearValueHead
from rl2.agents.integration.policy_net import StatefulPolicyNet
from rl2.agents.integration.value_net import StatefulValueNet
from rl2.algos.ppo import training_loop

from rl2.utils.checkpoint_util import maybe_load_checkpoint, save_checkpoint
from rl2.utils.comm_util import get_comm, sync_state
from rl2.utils.constants import ROOT_RANK
from rl2.utils.optim_util import get_weight_decay_param_groups
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

    env.new_env_fixed_target(target_x, target_y)        #create new env with fixed landing position

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
        env.env.render()
        #yield env.render()
        o_tp1, r_t, done_t, _ = env.step(a_t.squeeze(0).detach().numpy().item())
        o_t = np.array([o_tp1])
        a_tm1 = np.array([a_t.squeeze(0).detach().numpy()])
        r_tm1 = np.array([r_t])
        d_tm1 = np.array([float(done_t)])
<<<<<<< HEAD

def main():
    
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
            env.env.render()
            #yield env.render()
            o_tp1, r_t, done_t, _ = env.step(a_t.squeeze(0).detach().numpy().item())
            o_t = np.array([o_tp1])
            a_tm1 = np.array([a_t.squeeze(0).detach().numpy()])
            r_tm1 = np.array([r_t])
            d_tm1 = np.array([float(done_t)])
            h_tm1_policy_net = h_t_policy_net
    
    
    
    args = create_argparser().parse_args()
    comm = get_comm()
    
    env = create_env(
        environment=args.environment,
        num_states=args.num_states,
        num_actions=args.num_actions,
        max_episode_len=args.max_episode_len)
    
    # create learning system.
    policy_net = create_net(
        net_type='policy',
        environment=args.environment,
        architecture=args.architecture,
        num_states=args.num_states,
        num_actions=args.num_actions,
        num_features=args.num_features,
        context_size=args.meta_episode_len)

    
    # load checkpoint, if applicable.
    base_path = "checkpoints/defaults/policy_net"
    steps = 84
    model_path = os.path.join(base_path, f"model_{steps}.pth")
    policy_net.load_state_dict(tc.load(model_path, weights_only=True))
    
    evaluation_loop(
        env = env,
        policy_net = policy_net,
        target_x = 5,
        target_y = 5 
    )

if __name__ == "__main__":
    main()
=======
        h_tm1_policy_net = h_t_policy_net
>>>>>>> 4955cdef6411d9d55959fa36d3c4b9c7c619aa07
