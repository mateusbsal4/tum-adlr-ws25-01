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
import csv
import matplotlib.pyplot as plt


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
# from render_browser import render_browser




# @render_browser
@tc.no_grad()
def evaluation_loop(
        env: MetaEpisodicEnv,
        policy_net: StatefulPolicyNet,
        target_x: float,    #must be within [0, 1]!
        target_y: float,     #must be within [0, 1]!
        writer
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
    logging.info(f"fixed target: X,Y = ({target_x}, {target_y})")

    o_t = np.array([env.reset()])
    a_tm1 = np.array([0])
    r_tm1 = np.array([0.0])
    d_tm1 = np.array([1.0])
    h_tm1_policy_net = policy_net.initial_state(batch_size=1)
    done_t = 0
    timestep = 0
    while not done_t:
        pi_dist_t, h_t_policy_net = policy_net(
            curr_obs=tc.FloatTensor(o_t),
            prev_action=tc.LongTensor(a_tm1),
            prev_reward=tc.FloatTensor(r_tm1),
            prev_done=tc.FloatTensor(d_tm1),
            prev_state=h_tm1_policy_net)

        a_t = pi_dist_t.sample()
        env.env.render()
        # yield env.env.render()
        o_tp1, r_t, done_t, _ = env.step(a_t.squeeze(0).detach().numpy().item())
        o_t = np.array([o_tp1])
        a_tm1 = np.array([a_t.squeeze(0).detach().numpy()])
        r_tm1 = np.array([r_t])
        d_tm1 = np.array([float(done_t)])
        
        #save to csv
        writer.writerow([timestep, r_t])
        timestep += 1
        
    logging.info("Episode finished. Rewards saved to csv file.")
    print(f"Rewards saved to csv file.")
        

def main():
    
    
        
        
    # logging --------
    log_directory = 'checkpoints/logs/'
    os.makedirs(log_directory, exist_ok=True)
    log_filename = os.path.join(log_directory, 'eval_log.txt')
    logging.basicConfig(
        filename=log_filename,  # Output file where logs will be saved
        level=logging.INFO,           # Log level (INFO, DEBUG, etc.)
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
        filemode='a')                 # 'w' for writing (overwrites existing file), 'a' for appending
    print('start logging')
    logging.info("start logging")
    
    # Reward saving
    csv_file_path = os.path.join(log_directory, "eval_reward.csv")
    
    
    
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

    
    # load checkpoint.
    base_path = "checkpoints/defaults/policy_net"
    steps = 27
    model_path = os.path.join(base_path, f"model_{steps}.pth")
    policy_net.load_state_dict(tc.load(model_path, weights_only=True))
    
    print("evaluating")
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Episode', 'Reward'])  # Header
        evaluation_loop(
            env = env,
            policy_net = policy_net,
            target_x = 0, # between 0 and 1
            target_y = 0, # between 0 and 1
            writer = writer,
        )
    print("end evaluating")
    
    
    
    # Plotting the csv
    figure_file_path = os.path.join(log_directory, "eval_reward.png")

    # Read data from CSV
    timestep = []
    reward = []

    with open(csv_file_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header
        for row in reader:
            timestep.append(int(row[0]))
            reward.append(float(row[1]))
            
    print(f"Sum: {np.sum(reward)}, Mean: {np.mean(reward)}")

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(timestep, reward, marker="o", linestyle="-", color="r")
    plt.title("Reward during Evaluation Episode")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.grid()

    # Save the figure to a PNG file
    plt.savefig(figure_file_path, dpi=300)  # dpi=300 for high-resolution
    print(f"Plot saved as {figure_file_path}")

    # Optionally show the plot (can be omitted if not needed)
    # plt.show()

if __name__ == "__main__":
    main()