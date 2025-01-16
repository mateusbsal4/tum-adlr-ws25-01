"""
Implements ppo loss computations for training
stateful meta-reinforcement learning agents.
"""

import os
from typing import List, Dict, Optional, Callable
from collections import deque
import logging
import csv
from tqdm import tqdm

import torch as tc
import numpy as np
from mpi4py import MPI

from rl2.envs.abstract import MetaEpisodicEnv
from rl2.agents.integration.policy_net import StatefulPolicyNet
from rl2.agents.integration.value_net import StatefulValueNet
from rl2.algos.common import (
    MetaEpisode,
    generate_meta_episode,
    assign_credit,
    huber_func,
)
from rl2.utils.comm_util import sync_grads
from rl2.utils.constants import ROOT_RANK


def compute_losses(
        meta_episodes: List[MetaEpisode],
        policy_net: StatefulPolicyNet,
        value_net: StatefulValueNet,
        clip_param: float,
        ent_coef: float
    ) -> Dict[str, tc.Tensor]:
    """
    Computes the losses for Proximal Policy Optimization.

    Args:
        meta_episodes: list of meta-episodes.
        policy_net: policy network.
        value_net: value network.
        clip_param: clip parameter for PPO.
        ent_coef: entropy coefficient for PPO.

    Returns:
        loss_dict: a dictionary of losses.
    """
    def get_tensor(field, dtype=None):
        mb_field = np.stack(
            list(map(lambda metaep: getattr(metaep, field), meta_episodes)),
            axis=0)
        if dtype == 'long':
            return tc.LongTensor(mb_field)
        return tc.FloatTensor(mb_field)

    # minibatch data tensors
    mb_obs = get_tensor('obs', 'long')
    mb_acs = get_tensor('acs', 'long')
    mb_rews = get_tensor('rews')
    mb_dones = get_tensor('dones')
    mb_logpacs = get_tensor('logpacs')
    mb_advs = get_tensor('advs')
    mb_tdlam_rets = get_tensor('tdlam_rets')

    # input for loss calculations
    B = len(meta_episodes)
    ac_dummy = tc.zeros(dtype=tc.int64, size=(B,))
    rew_dummy = tc.zeros(dtype=tc.float32, size=(B,))
    done_dummy = tc.ones(dtype=tc.float32, size=(B,))

    curr_obs = mb_obs
    prev_action = tc.cat((ac_dummy.unsqueeze(1), mb_acs[:, 0:-1]), dim=1)
    prev_reward = tc.cat((rew_dummy.unsqueeze(1), mb_rews[:, 0:-1]), dim=1)
    prev_done = tc.cat((done_dummy.unsqueeze(1), mb_dones[:, 0:-1]), dim=1)
    prev_state_policy_net = policy_net.initial_state(batch_size=B)
    prev_state_value_net = value_net.initial_state(batch_size=B)

    # forward pass implements unroll for recurrent/attentive architectures.
    pi_dists, _ = policy_net(
        curr_obs=curr_obs,
        prev_action=prev_action,
        prev_reward=prev_reward,
        prev_done=prev_done,
        prev_state=prev_state_policy_net)

    vpreds, _ = value_net(
        curr_obs=curr_obs,
        prev_action=prev_action,
        prev_reward=prev_reward,
        prev_done=prev_done,
        prev_state=prev_state_value_net)

    entropies = pi_dists.entropy()
    logpacs_new = pi_dists.log_prob(mb_acs)
    vpreds_new = vpreds

    # entropy bonus
    meanent = tc.mean(entropies)
    policy_entropy_bonus = ent_coef * meanent

    # policy surrogate objective
    policy_ratios = tc.exp(logpacs_new - mb_logpacs)
    clipped_policy_ratios = tc.clip(policy_ratios, 1-clip_param, 1+clip_param)
    surr1 = mb_advs * policy_ratios
    surr2 = mb_advs * clipped_policy_ratios
    policy_surrogate_objective = tc.mean(tc.min(surr1, surr2))

    # composite policy loss
    policy_loss = -(policy_surrogate_objective + policy_entropy_bonus)

    # value loss
    value_loss = tc.mean(huber_func(mb_tdlam_rets, vpreds_new))

    # clipfrac
    clipfrac = tc.mean(tc.greater(surr1, surr2).float())

    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "meanent": meanent,
        "clipfrac": clipfrac
    }


def training_loop(
        env: MetaEpisodicEnv,
        policy_net: StatefulPolicyNet,
        value_net: StatefulValueNet,
        policy_optimizer: tc.optim.Optimizer,
        value_optimizer: tc.optim.Optimizer,
        policy_scheduler: Optional[tc.optim.lr_scheduler._LRScheduler],  # pylint: disable=W0212
        value_scheduler: Optional[tc.optim.lr_scheduler._LRScheduler],  # pylint: disable=W0212
        meta_episodes_per_policy_update: int,
        meta_episodes_per_learner_batch: int,
        meta_episode_len: int,
        ppo_opt_epochs: int,
        ppo_clip_param: float,
        ppo_ent_coef: float,
        discount_gamma: float,
        gae_lambda: float,
        standardize_advs: bool,
        max_pol_iters: int,
        pol_iters_so_far: int,
        policy_checkpoint_fn: Callable[[int], None],
        value_checkpoint_fn: Callable[[int], None],
        comm: type(MPI.COMM_WORLD),
    ) -> None:
    """
    Train a stateful RL^2 agent via PPO to maximize discounted cumulative reward
    in Tabular MDPs, sampled from the distribution used in Duan et al., 2016.

    Args:
        env: environment.
        policy_net: policy network.
        value_net: value network,
        policy_optimizer: policy optimizer.
        value_optimizer: value optimizer.
        policy_scheduler: policy lr scheduler.
        value_scheduler: value lr scheduler.
        meta_episodes_per_policy_update: meta-episodes per policy improvement,
            on each process.
        meta_episodes_per_learner_batch: meta-episodes per batch on each process.
        meta_episode_len: timesteps per meta-episode.
        ppo_opt_epochs: optimization epochs for proximal policy optimization.
        ppo_clip_param: clip parameter for proximal policy optimization.
        ppo_ent_coef: entropy bonus coefficient for proximal policy optimization
        discount_gamma: discount factor gamma.
        gae_lambda: decay parameter lambda for generalized advantage estimation.
        standardize_advs: standardize advantages to mean 0 and stddev 1?
        max_pol_iters: the maximum number policy improvements to make.
        pol_iters_so_far: the number of policy improvements made so far.
        policy_checkpoint_fn: a callback for saving checkpoints of policy net.
        value_checkpoint_fn: a callback for saving checkpoints of value net.
        comm: mpi comm_world communicator object.

    Returns:
        None
    """
    meta_ep_returns = deque(maxlen=1000)
    log_directory = 'checkpoints/logs/'
    show_pbar = True # optional: use progress bar to visualize the progress
    
 
    timestep = []
    reward = []

    for pol_iter in range(pol_iters_so_far, max_pol_iters):
        # create a new environment
        env.new_env()
        singlereward_csv_filepath = os.path.join(log_directory, f"politer{pol_iter}_reward.csv")
        
        # collect meta-episodes...
        meta_episodes = list()
        
        if comm.Get_rank() == ROOT_RANK and show_pbar:
            episode_pbar = tqdm(total=meta_episodes_per_policy_update,  desc="Collecting Meta-Episodes", ncols=80)
        
        
        for i in range(0, meta_episodes_per_policy_update): 
        # for i in range(0, meta_episodes_per_policy_update): 
            # collect one meta-episode and append it to the list
            meta_episode = generate_meta_episode(
                env=env,
                policy_net=policy_net,
                value_net=value_net,
                meta_episode_len=meta_episode_len)
            meta_episode = assign_credit(
                meta_episode=meta_episode,
                gamma=discount_gamma,
                lam=gae_lambda)
            meta_episodes.append(meta_episode)
            
            if comm.Get_rank() == ROOT_RANK and show_pbar:
                    episode_pbar.update(1)

            # logging
            l_meta_ep_returns = [np.sum(meta_episode.rews)] #local meta episode return from a single worker
            g_meta_ep_returns = comm.allgather(l_meta_ep_returns) # global meta episode return from all workers
            g_meta_ep_returns = [x for loc in g_meta_ep_returns for x in loc]
            meta_ep_returns.extend(g_meta_ep_returns) # all the rewards from different works in a list
            
            #save to csv 
            timestep.append(i)
            reward.append(meta_ep_returns)
                
        
            with open(singlereward_csv_filepath, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['Episode', 'Reward'])  # Header
                for i in range(len(timestep)):
                    #save to csv
                    writer.writerow([timestep[i], reward[i]])
                
        if comm.Get_rank() == ROOT_RANK and show_pbar:
            episode_pbar.close()

        # maybe standardize advantages...
        if standardize_advs:
            num_procs = comm.Get_size()
            adv_eps = 1e-8

            l_advs = list(map(lambda m: m.advs, meta_episodes))
            l_adv_mu = np.mean(l_advs)
            g_adv_mu = comm.allreduce(l_adv_mu, op=MPI.SUM) / num_procs

            l_advs_centered = list(map(lambda adv: adv - g_adv_mu, l_advs))
            l_adv_sigma2 = np.var(l_advs_centered)
            g_adv_sigma2 = comm.allreduce(l_adv_sigma2, op=MPI.SUM) / num_procs
            g_adv_sigma = np.sqrt(g_adv_sigma2) + adv_eps

            l_advs_standardized = list(map(lambda adv: adv / g_adv_sigma, l_advs_centered))
            for m, a in zip(meta_episodes, l_advs_standardized):
                setattr(m, 'advs', a)
                setattr(m, 'tdlam_rets', m.vpreds + a)

            if comm.Get_rank() == ROOT_RANK:
                mean_adv_r0 = np.mean(
                    list(map(lambda m: m.advs, meta_episodes)))
                logging.info(f"Mean advantage: {mean_adv_r0}")
                print(f"Mean advantage: {mean_adv_r0}")

        # update policy...
        for opt_epoch in range(ppo_opt_epochs):
            
            idxs = np.random.permutation(meta_episodes_per_policy_update)
            if comm.Get_rank() == ROOT_RANK:
                logging.info(f"pol update {pol_iter}, opt_epoch: {opt_epoch}...")
                print(f"\npol update {pol_iter}, opt_epoch: {opt_epoch}...\n")
                if show_pbar:
                    weight_pbar = tqdm(total=(meta_episodes_per_policy_update//meta_episodes_per_learner_batch), desc="Updating Weights")
                
            # optional: progress bar
            for i in range(0, meta_episodes_per_policy_update, meta_episodes_per_learner_batch):
            # for i in range(0, meta_episodes_per_policy_update, meta_episodes_per_learner_batch):
                mb_idxs = idxs[i:i+meta_episodes_per_learner_batch]
                mb_meta_eps = [meta_episodes[idx] for idx in mb_idxs]
                losses = compute_losses(
                    meta_episodes=mb_meta_eps,
                    policy_net=policy_net,
                    value_net=value_net,
                    clip_param=ppo_clip_param,
                    ent_coef=ppo_ent_coef)

                policy_optimizer.zero_grad()
                losses['policy_loss'].backward()
                sync_grads(model=policy_net, comm=comm)
                policy_optimizer.step()
                if policy_scheduler:
                    policy_scheduler.step()

                value_optimizer.zero_grad()
                losses['value_loss'].backward()
                sync_grads(model=value_net, comm=comm)
                value_optimizer.step()
                if value_scheduler:
                    value_scheduler.step()
                    
                if comm.Get_rank() == ROOT_RANK and show_pbar:
                    weight_pbar.update(1)

            
            if comm.Get_rank() == ROOT_RANK and show_pbar:
                weight_pbar.close()
            
            # logging
            global_losses = {}
            for name in losses:
                loss_sum = comm.allreduce(losses[name], op=MPI.SUM)
                loss_avg = loss_sum / comm.Get_size()
                global_losses[name] = loss_avg

            if comm.Get_rank() == ROOT_RANK:
                for name, value in global_losses.items():
                    logging.info(f"\t{name}: {value:>0.6f}")
                    print(f"\t{name}: {value:>0.6f}")

            
        
        # misc.: print metrics, save checkpoint.
        if comm.Get_rank() == ROOT_RANK:
            logging.info("-" * 100)
            logging.info(f"mean meta-episode return: {np.mean(meta_ep_returns):>0.3f}")
            logging.info("-" * 100)
            print("-" * 100)
            print(f"mean meta-episode return: {np.mean(meta_ep_returns):>0.3f}")
            print("-" * 100)
            policy_checkpoint_fn(pol_iter + 1)
            value_checkpoint_fn(pol_iter + 1)
            
            
    
    
