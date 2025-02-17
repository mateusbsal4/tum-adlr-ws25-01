"""
Script for training stateful meta-reinforcement learning agents
"""

import argparse
from functools import partial
import logging
import os
import wandb  # Add this import
#from render_browser import render_browser
import torch as tc
import numpy as np
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


def create_argparser():
    parser = argparse.ArgumentParser(
        description="""Training script for RL^2.""")

    ### Environment
    parser.add_argument("--environment", choices=['lander', 'bandit', 'tabular_mdp'],
                        default='lander')
    parser.add_argument("--num_states", type=int, default=8,
                        help="Ignored if environment is bandit.")
    parser.add_argument("--num_actions", type=int, default=4)
    parser.add_argument("--max_episode_len", type=int, default=1_000,  # apparently completely irrelevant (not used anywhere)
                        help="Timesteps before automatic episode reset. " +
                             "Ignored if environment is bandit.")
    parser.add_argument("--meta_episode_len", type=int, default=1_000, 
                        help="Timesteps per meta-episode.")

    ### Architecture
    parser.add_argument(
        "--architecture", choices=['gru', 'lstm', 'snail', 'transformer'],
        default='lstm')
    parser.add_argument("--num_features", type=int, default=128) # num of features in the hidden layer

    ### Checkpointing
    parser.add_argument("--model_name", type=str, default='defaults')
    parser.add_argument("--checkpoint_dir", type=str, default='checkpoints')

    ### Training
    parser.add_argument("--max_pol_iters", type=int, default=5000) # like vanilla lander
    parser.add_argument("--timesteps_per_pol_update", type=int, default=2048,
                        help="If -1, quantity is determined using a formula")
    parser.add_argument("--timesteps_per_learner_batch", type=int, default=64) #  vanilla PPO = 64 timesteps
    parser.add_argument("--meta_ep_per_learner_batch", type=int, default=10) #  vanilla PPO = 64 timesteps
    parser.add_argument("--ppo_opt_epochs", type=int, default=10)
    
    parser.add_argument("--ppo_clip_param", type=float, default=0.1118520486242931) # like vanilla lander
    parser.add_argument("--ppo_ent_coef", type=float, default=0.03817033721208064) #adjusted for LSTM
    parser.add_argument("--discount_gamma", type=float, default=0.9839244506585908)
    parser.add_argument("--gae_lambda", type=float, default=0.9719123597888648) # like vanilla lander
    parser.add_argument("--standardize_advs", type=int, choices=[0,1], default=1) # adjusted for LSTM
    parser.add_argument("--adam_lr", type=float, default=0.0009931916739343528)
    parser.add_argument("--adam_eps", type=float, default=1e-5)
    parser.add_argument("--adam_wd", type=float, default=0.01)
    parser.add_argument("--value_lr", type=float, default=0.000594249166404675)
    parser.add_argument("--wandb_project", type=str, default='rl2_project')  # Add this line
    parser.add_argument("--wandb_entity", type=str, default='')  # Add this line
    parser.add_argument("--value_loss_threshold", type=float, default=1_000, help="Threshold for value loss to abort the run")  # Add this line
    parser.add_argument("--visualize_weights", action="store_true", default=False, help="Toggle weight visualization during training")
    return parser


def create_env(environment, num_states, num_actions, max_episode_len):
    if environment == 'lander':
        return LunarLanderEnv(max_episode_length = max_episode_len)
    if environment == 'bandit':
        return BanditEnv(
            num_actions=num_actions)
    if environment == 'tabular_mdp':
        return MDPEnv(
            num_states=num_states,
            num_actions=num_actions,
            max_episode_length=max_episode_len)
    raise NotImplementedError


def create_preprocessing(environment, num_states, num_actions):
    if environment == 'lander':
        return LunarLanderPreprocessing(num_actions = num_actions)
    if environment == 'bandit':
        return MABPreprocessing(
            num_actions=num_actions)
    if environment == 'tabular_mdp':
        return MDPPreprocessing(
            num_states=num_states,
            num_actions=num_actions)
    raise NotImplementedError


def create_architecture(architecture, input_dim, num_features, context_size):
    if architecture == 'gru':
        return GRU(
            input_dim=input_dim,
            hidden_dim=num_features,
            forget_bias=1.0,
            use_ln=True,
            reset_after=True)
    if architecture == 'lstm':
        return LSTM(
            input_dim=input_dim,
            hidden_dim=num_features,
            forget_bias=1.0,
            use_ln=True)
    if architecture == 'snail':
        return SNAIL(
            input_dim=input_dim,
            feature_dim=num_features,
            context_size=context_size,
            use_ln=True)
    if architecture == 'transformer':
        return Transformer(
            input_dim=input_dim,
            feature_dim=num_features,
            n_layer=9,
            n_head=2,
            n_context=context_size)
    raise NotImplementedError


def create_head(head_type, num_features, num_actions):
    if head_type == 'policy':
        return LinearPolicyHead(
            num_features=num_features,
            num_actions=num_actions)
    if head_type == 'value':
        return LinearValueHead(
            num_features=num_features)
    raise NotImplementedError


def create_net(
        net_type, environment, architecture, num_states, num_actions,
        num_features, context_size
):
    preprocessing = create_preprocessing(
        environment=environment,
        num_states=num_states,
        num_actions=num_actions)
    architecture = create_architecture(
        architecture=architecture,
        input_dim=preprocessing.output_dim,
        num_features=num_features,
        context_size=context_size)
    head = create_head(
        head_type=net_type,
        num_features=architecture.output_dim,
        num_actions=num_actions)

    if net_type == 'policy':
        return StatefulPolicyNet(
            preprocessing=preprocessing,
            architecture=architecture,
            policy_head=head)
    if net_type == 'value':
        return StatefulValueNet(
            preprocessing=preprocessing,
            architecture=architecture,
            value_head=head)
    raise NotImplementedError


# @render_browser
def main():
        
    args = create_argparser().parse_args()
    comm = get_comm()

    # Initialize wandb
    # wandb.init(project=args.wandb_project, entity=args.wandb_entity)  # Add this line

    # logging --------
    log_directory = os.path.join(args.checkpoint_dir,"logs")
    os.makedirs(log_directory, exist_ok=True)
    log_filename = os.path.join(log_directory, 'training_log.txt')
    logging.basicConfig(
        filename=log_filename,  # Output file where logs will be saved
        level=logging.INFO,           # Log level (INFO, DEBUG, etc.)
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
        filemode='a')                 # 'w' for writing (overwrites existing file), 'a' for appending
    print('start logging')
    logging.info("start logging")
    
    # Log all arguments
    logging.info("Command line arguments:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    logging.info("")

    # create env.
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

    value_net = create_net(
        net_type='value',
        environment=args.environment,
        architecture=args.architecture,
        num_states=args.num_states,
        num_actions=args.num_actions,
        num_features=args.num_features,
        context_size=args.meta_episode_len)

    policy_optimizer = tc.optim.AdamW(
        get_weight_decay_param_groups(policy_net, args.adam_wd),
        lr=args.adam_lr,
        eps=args.adam_eps)
    value_optimizer = tc.optim.AdamW(
        get_weight_decay_param_groups(value_net, args.adam_wd),
        lr=args.value_lr,  # Use value_lr here
        eps=args.adam_eps)

    # Add learning rate schedulers
    policy_scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(
        policy_optimizer, 
        mode='max',  # We want to maximize reward
        factor=0.5,  # Reduce LR by half when plateauing
        patience=5,
        verbose=True,
        threshold=1.0,  # Minimum change in the monitored quantity to qualify as an improvement
        threshold_mode='rel'  # The threshold is relative to best value
    )
    
    value_scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(
        value_optimizer,
        mode='min',  # We want to minimize value loss
        factor=0.5,
        patience=3,
        verbose=True,
        threshold=0.01,  # Minimum change in the monitored quantity to qualify as an improvement
        threshold_mode='rel'  # The threshold is relative to best value
    )

    # load checkpoint, if applicable.
    pol_iters_so_far = 0
    if comm.Get_rank() == ROOT_RANK:
        a = maybe_load_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            model_name=f"{args.model_name}/policy_net",
            model=policy_net,
            optimizer=policy_optimizer,
            scheduler=policy_scheduler,
            steps=None)

        b = maybe_load_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            model_name=f"{args.model_name}/value_net",
            model=value_net,
            optimizer=value_optimizer,
            scheduler=value_scheduler,
            steps=None)

        if a != b:
            raise RuntimeError(
                "Policy and value iterates not aligned in latest checkpoint!")
        pol_iters_so_far = a

    # sync state.
    pol_iters_so_far = comm.bcast(pol_iters_so_far, root=ROOT_RANK)
    sync_state(
        model=policy_net,
        optimizer=policy_optimizer,
        scheduler=policy_scheduler,
        comm=comm,
        root=ROOT_RANK)
    sync_state(
        model=value_net,
        optimizer=value_optimizer,
        scheduler=value_scheduler,
        comm=comm,
        root=ROOT_RANK)

    # make callback functions for checkpointing.
    policy_checkpoint_fn = partial(
        save_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        model_name=f"{args.model_name}/policy_net",
        model=policy_net,
        optimizer=policy_optimizer,
        scheduler=policy_scheduler)

    value_checkpoint_fn = partial(
        save_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        model_name=f"{args.model_name}/value_net",
        model=value_net,
        optimizer=value_optimizer,
        scheduler=value_scheduler)

    # run it!
    if args.timesteps_per_pol_update == -1:
        numer = 2048 # timesteps per policy update
        denom = comm.Get_size() * args.meta_episode_len
        timesteps_per_pol_update = numer // denom
    else:
        timesteps_per_pol_update = args.timesteps_per_pol_update // comm.Get_size()

    
    training_loop(
        env=env,
        policy_net=policy_net,
        value_net=value_net,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        policy_scheduler=policy_scheduler,
        value_scheduler=value_scheduler,
        timesteps_per_pol_update=timesteps_per_pol_update,
        meta_ep_per_learner_batch=args.meta_ep_per_learner_batch,
        meta_episode_len=args.meta_episode_len,
        ppo_opt_epochs=args.ppo_opt_epochs,
        ppo_clip_param=args.ppo_clip_param,
        ppo_ent_coef=args.ppo_ent_coef,
        discount_gamma=args.discount_gamma,
        gae_lambda=args.gae_lambda,
        standardize_advs=bool(args.standardize_advs),
        max_pol_iters=args.max_pol_iters,
        pol_iters_so_far=pol_iters_so_far,
        policy_checkpoint_fn=policy_checkpoint_fn,
        value_checkpoint_fn=value_checkpoint_fn,
        checkpoint_dir=args.checkpoint_dir,
        comm=comm,
        value_loss_threshold=args.value_loss_threshold,
        visualize_weights_flag=args.visualize_weights
    )
    
    print("Training Ended!")
    logging.info("Training Ended!")

if __name__ == '__main__':
    main()
