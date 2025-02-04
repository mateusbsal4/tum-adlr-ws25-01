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
import matplotlib.pyplot as plt

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
from datetime import datetime
import wandb  # Add this import



def compute_losses(
    obs: np.ndarray,
    acs: np.ndarray,
    rews: np.ndarray,
    dones: np.ndarray,
    logpacs_old: np.ndarray,
    advs: np.ndarray,
    tdlam_rets: np.ndarray,
    policy_net: StatefulPolicyNet,
    value_net: StatefulValueNet,
    clip_param: float,
    ent_coef: float
    ):
    """
    Computes PPO losses given flattened arrays for timesteps.

    obs: shape [N, obs_dim]
    acs, rews, dones, logpacs_old, advs, tdlam_rets: shape [N]
    """

    # Convert to tensors
    B = len(rews)
    t_obs = tc.FloatTensor(obs)                   # [N, obs_dim]
    t_acs = tc.LongTensor(acs)                    # [N]
    t_rews = tc.FloatTensor(rews)                 # [N]
    t_dones = tc.FloatTensor(dones)               # [N]
    t_logpacs_old = tc.FloatTensor(logpacs_old)   # [N]
    t_advs = tc.FloatTensor(advs)                 # [N]
    t_tdlam_rets = tc.FloatTensor(tdlam_rets)     # [N]
    prev_state_policy_net = policy_net.initial_state(batch_size=B)
    prev_state_value_net = value_net.initial_state(batch_size=B)

    # Forward through feed-forward policy & value
    pi_dists, _ = policy_net(
        curr_obs=t_obs,
        prev_action=t_acs,
        prev_reward=t_rews,
        prev_done=t_dones,
        prev_state=prev_state_policy_net) # distribution over actions, shape [N, ...]
    
    vpreds, _ = value_net(
        curr_obs=t_obs,
        prev_action=t_acs,
        prev_reward=t_rews,
        prev_done=t_dones,
        prev_state=prev_state_value_net)          # predicted values, shape [N]
    # If your nets are originally RNN-based, you'd need to handle sequence states carefully.

    # Entropy
    entropies = pi_dists.entropy()     # shape [N]
    vpreds_new = vpreds
    
    meanent = entropies.mean()
    policy_entropy_bonus = ent_coef * meanent

    # New logp
    logpacs_new = pi_dists.log_prob(t_acs)  # shape [N]
    
    # Normalize advantages
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    
    # PPO ratio with clipping for numerical stability
    ratio = tc.exp(tc.clamp(logpacs_new - t_logpacs_old, -20, 20))
    surr1 = ratio * t_advs
    surr2 = tc.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * t_advs
    
    # Policy loss with additional clipping
    policy_surrogate_objective = tc.mean(tc.min(surr1, surr2))
    policy_loss = -(tc.clamp(policy_surrogate_objective, -20.0, 20.0) + policy_entropy_bonus)
    
    # Value loss with normalization
    target_values = (t_tdlam_rets - t_tdlam_rets.mean()) / (t_tdlam_rets.std() + 1e-8)
    predicted_values = (vpreds_new - vpreds_new.mean()) / (vpreds_new.std() + 1e-8)
    value_loss = tc.nn.MSELoss()(predicted_values, target_values)
    
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
        timesteps_per_pol_update: int,
        timesteps_per_learner_batch: int,
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
        checkpoint_dir: str,
        comm: type(MPI.COMM_WORLD),
        value_loss_threshold: float,  # Add this parameter
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
        timesteps_per_pol_update: meta-episodes per policy improvement,
            on each process.
        timesteps_per_learner_batch: meta-episodes per batch on each process.
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
        value_loss_threshold: threshold for value loss to abort the run.

    Returns:
        None
    """
    

    def flatten_episodes(meta_episodes):
        """
        Flatten a list of MetaEpisode into arrays of shape [N, ...],
        where N is the total number of timesteps across all episodes.
        """
        obs_list = []
        acs_list = []
        rews_list = []
        dones_list = []
        logpacs_list = []
        advs_list = []
        tdlam_rets_list = []

        for ep in meta_episodes:
            obs_list.append(ep.obs)           # shape [T, obs_dim]
            acs_list.append(ep.acs)           # shape [T]
            rews_list.append(ep.rews)         # shape [T]
            dones_list.append(ep.dones)       # shape [T]
            logpacs_list.append(ep.logpacs)   # shape [T]
            advs_list.append(ep.advs)         # shape [T]
            tdlam_rets_list.append(ep.tdlam_rets)  # shape [T]

        # Concatenate along time dimension
        obs = np.concatenate(obs_list, axis=0)         # [N, obs_dim]
        acs = np.concatenate(acs_list, axis=0)         # [N]
        rews = np.concatenate(rews_list, axis=0)       # [N]
        dones = np.concatenate(dones_list, axis=0)     # [N]
        logpacs = np.concatenate(logpacs_list, axis=0) # [N]
        advs = np.concatenate(advs_list, axis=0)       # [N]
        tdlam_rets = np.concatenate(tdlam_rets_list, axis=0)  # [N]

        return (
            obs, acs, rews, dones, logpacs, advs, tdlam_rets
            )


    meta_ep_returns = deque(maxlen=1000) 
    log_directory = os.path.join(checkpoint_dir, 'logs')
    os.makedirs(log_directory, exist_ok=True)
    show_pbar = False # optional: use progress bar to visualize the progress
    
 
    policy_losses = []
    value_losses = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize the progress bar for timesteps_till_now
    if comm.Get_rank() == ROOT_RANK and show_pbar:
        total_timesteps_pbar = tqdm(total=max_pol_iters * 2048, desc="Total Timesteps", position=2, ncols=80, leave=True)

    # Add moving averages for monitoring
    reward_window = deque(maxlen=100)
    value_loss_window = deque(maxlen=100)
    
    # Early stopping variables
    best_loss = float('+inf')
    patience = 10
    patience_counter = 0
    
    for pol_iter in range(pol_iters_so_far, max_pol_iters):
        # create a new environment
        env.new_env()
        timesteps_till_now = 0
        local_timesteps = 0
        total_timesteps = 0
        
        # collect meta-episodes...
        meta_episodes = list()
        
        if comm.Get_rank() == ROOT_RANK and show_pbar:
            episode_pbar = tqdm(total=timesteps_per_pol_update, desc="Collecting Meta-Episodes", ncols=80, position=0)
        
        successful_episodes = 0
        failed_episodes = 0
        
        local_episode_count = 0
        

        
        while total_timesteps < timesteps_per_pol_update:
            
            # collect one meta-episode and append it to the list
            meta_episode = generate_meta_episode(
                env=env,
                policy_net=policy_net,
                value_net=value_net,
                )
            meta_episode = assign_credit(
                meta_episode=meta_episode,
                gamma=discount_gamma,
                lam=gae_lambda)
            meta_episodes.append(meta_episode)
            
            
            # check if episode was successful or failed
            rewards = meta_episode.rews  # Get the last reward of the episode
            last_reward = rewards[-1]
            
            local_episode_count += 1
            episode_count = comm.allreduce(local_episode_count, op=MPI.SUM)
            logging.info(f"pol iter {pol_iter}, meta-episode {local_episode_count}")
            
            local_timesteps = len(rewards)
            global_timesteps = comm.allreduce(local_timesteps, op=MPI.SUM)
            total_timesteps += global_timesteps
            timesteps_till_now += global_timesteps

            if comm.Get_rank() == ROOT_RANK and show_pbar:
                total_timesteps_pbar.update(global_timesteps)
            
            if comm.Get_rank() == ROOT_RANK and show_pbar:
                    episode_pbar.update(global_timesteps)

            
            # local success and failure counts
            local_successful_episodes = 1 if last_reward == 100 else 0
            local_failed_episodes = 1 if last_reward == -100 else 0

            # gather success and failure counts across workers
            successful_episodes += comm.allreduce(local_successful_episodes, op=MPI.SUM)
            failed_episodes += comm.allreduce(local_failed_episodes, op=MPI.SUM)

                        
            # logging
            l_meta_ep_returns = [np.sum(meta_episode.rews)] #local meta episode return from a single worker
            # print(episode_count, ": ", l_meta_ep_returns, "epsiode length: ", len(meta_episode.rews))
            
            if np.sum(meta_episode.rews) > 50:
                print(episode_count, " episode - positive reward: ", l_meta_ep_returns, " last reward: ", last_reward)
                
                shaping = (
                    -100 * np.sqrt(meta_episode.obs[:,0] * meta_episode.obs[:,0] + meta_episode.obs[:,1] * meta_episode.obs[:,1])
                    - 100 * np.sqrt(meta_episode.obs[:,2] * meta_episode.obs[:,2] + meta_episode.obs[:,3] * meta_episode.obs[:,3])
                    - 100 * abs(meta_episode.obs[:,4])
                    + 10 * meta_episode.obs[:,6]
                    + 10 * meta_episode.obs[:,7]
                ) 
                
                plt.figure(figsize=(10, 6))
                plt.plot(-100 * np.sqrt(meta_episode.obs[:,0]**2 + meta_episode.obs[:,1]**2), linestyle='-', label='pos_reward')
                plt.plot(-100 * np.sqrt(meta_episode.obs[:,2]**2 + meta_episode.obs[:,3]**2), linestyle='-', label='vel_reward')
                plt.plot(-100 * abs(meta_episode.obs[:,4]), linestyle='-', label='angle_reward')
                plt.plot(10 * meta_episode.obs[:,5], linestyle='-', label='leg1_reward')
                plt.plot(10 * meta_episode.obs[:,6], linestyle='-', label='leg2_reward')
                plt.plot(shaping, linestyle='-', label='calculated_reward')
                plt.plot(meta_episode.rews, linestyle='-', label='Total Reward')
                plt.plot(meta_episode.acs, linestyle='-', label='Actions')
                plt.title(f'Episode {episode_count} - Rewards (Sum: {np.sum(meta_episode.rews):.2f}) within {len(meta_episode.rews)} steps)')
                plt.xlabel('Reward Index')
                plt.ylabel('Reward Value')
                plt.legend()
                plt.grid(True)
                filename = os.path.join(log_directory, f'pol_{pol_iter}_ep_{episode_count}_rewcomps_{timestamp}.png')
                plt.savefig(filename)
                plt.close()
                
            
            g_meta_ep_returns = comm.allgather(l_meta_ep_returns) # global meta episode return from all workers
            g_meta_ep_returns = [x for loc in g_meta_ep_returns for x in loc]
            meta_ep_returns.extend(g_meta_ep_returns) # all the rewards from different works in a list
            
            
        print("\nfinished collecting after ", total_timesteps, " timesteps")
        

        print(f"pol iter {pol_iter}: {successful_episodes} successes, {failed_episodes} failures, out of {episode_count} episodes")

                
        if comm.Get_rank() == ROOT_RANK and show_pbar:
            episode_pbar.close()

        # maybe standardize advantages...
        if standardize_advs:
            num_procs = comm.Get_size()
            adv_eps = 1e-8

            # Flatten all advantages
            all_advs = np.concatenate([m.advs for m in meta_episodes])

            # Compute global mean and std
            l_adv_mu = np.mean(all_advs)
            g_adv_mu = comm.allreduce(l_adv_mu, op=MPI.SUM) / num_procs

            l_advs_centered = all_advs - g_adv_mu
            l_adv_sigma2 = np.var(l_advs_centered)
            g_adv_sigma2 = comm.allreduce(l_adv_sigma2, op=MPI.SUM) / num_procs
            g_adv_sigma = np.sqrt(g_adv_sigma2) + adv_eps

            # Standardize advantages
            all_advs_standardized = l_advs_centered / g_adv_sigma

            # Split back into meta-episodes
            start_idx = 0
            for m in meta_episodes:
                end_idx = start_idx + len(m.advs)
                m.advs = all_advs_standardized[start_idx:end_idx]
                m.tdlam_rets = m.vpreds + m.advs
                start_idx = end_idx

            if comm.Get_rank() == ROOT_RANK:
                mean_adv_r0 = np.mean(all_advs_standardized)
                logging.info(f"Mean advantage: {mean_adv_r0}")
                print(f"Mean advantage: {mean_adv_r0}")


        # Suppose you've just collected `meta_episodes` via generate_meta_episode, etc.
        # Flatten them into a single buffer of timesteps
        (
            obs,
            acs,
            rews,
            dones,
            logpacs,
            advs,
            tdlam_rets
        ) = flatten_episodes(meta_episodes)

        N = acs.shape[0]  # total timesteps collected
        indices = np.arange(N)

        for opt_epoch in range(ppo_opt_epochs):
            # Shuffle all timesteps
            np.random.shuffle(indices)
            
            batch_size = timesteps_per_learner_batch

            if comm.Get_rank() == ROOT_RANK:
                logging.info(f"pol update {pol_iter}, opt_epoch: {opt_epoch}...")
                print(f"\npol update {pol_iter}, opt_epoch: {opt_epoch}...\n")
                if show_pbar:
                    # total number of minibatches
                    total_mb = (N // batch_size)
                    weight_pbar = tqdm(total=total_mb, desc="Updating Weights")

            # Slice minibatches of timesteps
            for start_idx in range(0, N, batch_size):
                end_idx = start_idx + batch_size
                mb_idxs = indices[start_idx:end_idx]

                # Extract the minibatch from flattened arrays
                mb_obs = obs[mb_idxs]
                mb_acs = acs[mb_idxs]
                mb_rews = rews[mb_idxs]
                mb_dones = dones[mb_idxs]
                mb_logpacs = logpacs[mb_idxs]
                mb_advs = advs[mb_idxs]
                mb_tdlam_rets = tdlam_rets[mb_idxs]
                
                # Compute losses on this batch
                losses = compute_losses(
                    obs=mb_obs,
                    acs=mb_acs,
                    rews=mb_rews,
                    dones=mb_dones,
                    logpacs_old=mb_logpacs,
                    advs=mb_advs,
                    tdlam_rets=mb_tdlam_rets,
                    policy_net=policy_net,
                    value_net=value_net,
                    clip_param=ppo_clip_param,
                    ent_coef=ppo_ent_coef
                )

                # Backprop: Policy
                policy_optimizer.zero_grad()
                losses['policy_loss'].backward()
                sync_grads(model=policy_net, comm=comm)

                # Add gradient clipping for policy network
                for param in policy_net.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-1, 1)  # Clip gradients between -1 and 1

                policy_optimizer.step()
                if policy_scheduler:
                    # Use mean reward as metric for policy scheduler
                    policy_scheduler.step(np.mean(meta_ep_returns))

                # Backprop: Value
                value_optimizer.zero_grad()
                losses['value_loss'].backward()
                sync_grads(model=value_net, comm=comm)
                
                # Gradient clipping for value network   
                for param in value_net.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-1, 1)
                value_optimizer.step()
                if value_scheduler:
                    # Use value loss as metric for value scheduler
                    value_scheduler.step(losses['value_loss'].item())

                if comm.Get_rank() == ROOT_RANK and show_pbar:
                    weight_pbar.update(1)
                
                

            if comm.Get_rank() == ROOT_RANK and show_pbar:
                weight_pbar.close()

            # Logging final losses from last minibatch
            global_losses = {}
            for name in losses:
                loss_sum = comm.allreduce(losses[name], op=MPI.SUM)
                loss_avg = loss_sum / comm.Get_size()
                global_losses[name] = loss_avg

            # Log losses to wandb
            # wandb.log({
            #     "policy_loss": global_losses['policy_loss'].item(),
            #     "value_loss": global_losses['value_loss'].item(),
            #     "meanent": global_losses['meanent'].item(),
            #     "clipfrac": global_losses['clipfrac'].item(),
            #     "mean reward": np.mean(meta_ep_returns),
            # })

            # Collect losses for plotting
            policy_losses.append(global_losses['policy_loss'].detach().numpy())
            value_losses.append(global_losses['value_loss'].detach().numpy())

            if comm.Get_rank() == ROOT_RANK:
                for name, value in global_losses.items():
                    logging.info(f"\t{name}: {value:>0.6f}")
                    print(f"\t{name}: {value:>0.6f}")
                    if name == 'value_loss' and value > value_loss_threshold:
                        logging.info(f"Value loss {value} exceeds threshold {value_loss_threshold}. Aborting run.")
                        print(f"Value loss {value} exceeds threshold {value_loss_threshold}. Aborting run.")
                        # wandb.run.finish()  # Abort the run
                        return  # Exit the training loop
            
            # EARLY STOPPING ---------
            # Monitor metrics
            # mean_reward = np.mean(meta_ep_returns)
            # reward_window.append(mean_reward)
            # current_loss = global_losses['value_loss'].item()
            # value_loss_window.append(current_loss)
            
            # # Early stopping check
            # if current_loss < best_loss:
            #     best_loss = current_loss
            #     patience_counter = 0
            # else:
            #     patience_counter += 1
            
            # if patience_counter >= patience:
            #     print("Early stopping triggered")
            #     break
            
            # # Abort if value loss is too high
            # if np.mean(value_loss_window) > value_loss_threshold:
            #     print("Value loss too high, aborting training")
            #     break
            
            
        # Plot and save the policy losses
        if comm.Get_rank() == ROOT_RANK:
            fig, axs = plt.subplots(2, 1, figsize=(10, 12))

            axs[0].plot(policy_losses, label='Policy Loss')
            axs[0].set_xlabel('Epoch')
            axs[0].set_ylabel('Policy Loss')
            axs[0].set_title('Policy Loss Across Epochs')
            axs[0].legend()
            axs[0].grid(True)

            axs[1].plot(value_losses, label='Value Loss')
            axs[1].set_xlabel('Epoch')
            axs[1].set_ylabel('Value Loss')
            axs[1].set_title('Value Loss Across Epochs')
            axs[1].legend()
            axs[1].grid(True)

            fig.tight_layout()
            loss_plot_filename = os.path.join(log_directory, f'losses_{timestamp}.png')
            plt.savefig(loss_plot_filename)
            plt.close()

            
        
        # misc.: print metrics, save checkpoint.
        if comm.Get_rank() == ROOT_RANK:
            logging.info("-" * 100)
            logging.info(f"pol iter: {pol_iter} - mean meta-episode return: {np.mean(meta_ep_returns):>0.3f}")
            logging.info("-" * 100)
            print("-" * 80)
            print(f"pol iter: {pol_iter} - mean meta-episode return: {np.mean(meta_ep_returns):>0.3f}")
            print(f"mean episode length: {total_timesteps/episode_count:.2f}")
            print("-" * 80)
            policy_checkpoint_fn(pol_iter + 1)
            value_checkpoint_fn(pol_iter + 1)
