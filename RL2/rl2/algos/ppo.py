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
import wandb
from rl2.utils.log2csv import log2csv
from rl2.utils.visualize_weights import visualize_weights
from torch.utils.tensorboard import SummaryWriter

def compute_losses(
    meta_episodes,
    policy_net: StatefulPolicyNet,
    value_net: StatefulValueNet,
    clip_param: float,
    ent_coef: float,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.015
) -> Dict[str, tc.Tensor]:
    """
    Computes PPO losses treating each episode as a sequence.
    
    Args:
        meta_episodes: List of MetaEpisode objects
        policy_net: Policy network
        value_net: Value network
        clip_param: PPO clip parameter
        ent_coef: Entropy bonus coefficient
        max_grad_norm: Maximum gradient norm for clipping
        target_kl: Target KL divergence for early stopping
    """
    if not meta_episodes:
        raise ValueError("Empty meta_episodes list provided")

    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    total_approx_kl = 0
    num_episodes = len(meta_episodes)

    # First pass: collect all advantages for normalization
    all_advs = []
    for episode in meta_episodes:
        all_advs.extend(episode.advs)
    all_advs = tc.FloatTensor(all_advs)
    
    # Compute advantage statistics
    adv_mean = all_advs.mean()
    adv_std = all_advs.std() + 1e-8  # Add epsilon for numerical stability

    for episode in meta_episodes:
        try:
            # Convert episode data to tensors and move to appropriate device
            device = next(policy_net.parameters()).device
            t_obs = tc.FloatTensor(episode.obs).to(device)
            t_acs = tc.LongTensor(episode.acs).to(device)
            t_rews = tc.FloatTensor(episode.rews).to(device)
            t_dones = tc.FloatTensor(episode.dones).to(device)
            t_logpacs_old = tc.FloatTensor(episode.logpacs).to(device)
            t_advs = tc.FloatTensor(episode.advs).to(device)
            t_tdlam_rets = tc.FloatTensor(episode.tdlam_rets).to(device)

            # Normalize advantages
            t_advs = (t_advs - adv_mean) / adv_std

            # Initialize LSTM states
            h_policy = policy_net.initial_state(batch_size=1)
            h_value = value_net.initial_state(batch_size=1)

            # Forward pass through the entire sequence
            pi_dists = []
            vpreds = []
            
            # Validate inputs
            if tc.isnan(t_obs).any() or tc.isinf(t_obs).any():
                raise ValueError("NaN or Inf detected in observations")
            
            for t in range(len(episode.obs)):
                # Get current timestep data
                obs_t = t_obs[t:t+1]
                act_t = t_acs[t:t+1]
                rew_t = t_rews[t:t+1]
                done_t = t_dones[t:t+1]

                # Policy network forward pass
                pi_dist_t, h_policy = policy_net(
                    curr_obs=obs_t,
                    prev_action=act_t,
                    prev_reward=rew_t,
                    prev_done=done_t,
                    prev_state=h_policy
                )
                pi_dists.append(pi_dist_t)

                # Value network forward pass
                vpred_t, h_value = value_net(
                    curr_obs=obs_t,
                    prev_action=act_t,
                    prev_reward=rew_t,
                    prev_done=done_t,
                    prev_state=h_value
                )
                vpreds.append(vpred_t)

            # Stack predictions
            logpacs_new = tc.stack([dist.log_prob(act) 
                                for dist, act in zip(pi_dists, t_acs)])
            vpreds_new = tc.cat(vpreds)
            
            # Dynamic ratio clipping based on policy distribution
            ratio = tc.exp(tc.clamp(logpacs_new - t_logpacs_old, 
                                -2.0 * tc.std(t_logpacs_old),
                                2.0 * tc.std(t_logpacs_old)))
            
            # Compute approximate KL divergence
            approx_kl = ((ratio - 1) - tc.log(ratio)).mean()
            total_approx_kl += approx_kl

            # Policy loss with clipping
            surr1 = ratio * t_advs
            surr2 = tc.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * t_advs
            policy_loss = -tc.mean(tc.min(surr1, surr2))

            # Add entropy bonus
            entropy = tc.mean(tc.stack([dist.entropy() for dist in pi_dists]))
            policy_loss = policy_loss - ent_coef * entropy

            # Compute value loss with Huber loss
            value_loss = huber_func(vpreds_new, t_tdlam_rets)

            # Gradient clipping
            if max_grad_norm > 0:
                tc.nn.utils.clip_grad_norm_(
                    list(policy_net.parameters()) + list(value_net.parameters()),
                    max_grad_norm
                )

            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_entropy += entropy

        except Exception as e:
            logging.error(f"Error processing episode: {str(e)}")
            continue

    # Average losses across episodes
    avg_approx_kl = total_approx_kl / num_episodes

    return {
        "policy_loss": total_policy_loss / num_episodes,
        "value_loss": total_value_loss / num_episodes,
        "meanent": total_entropy / num_episodes,
        "approx_kl": avg_approx_kl,
        "stop_training": avg_approx_kl > target_kl
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
        meta_ep_per_learner_batch: int,
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
        visualize_weights_flag: bool = False
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
        visualize_weights_flag: flag to control visualization

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
    global_update_step = 0
    
 
    policy_losses = []
    value_losses = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize the progress bar for timesteps_till_now
    if comm.Get_rank() == ROOT_RANK and show_pbar:
        total_timesteps_pbar = tqdm(total=max_pol_iters * 2048, desc="Total Timesteps", position=2, ncols=80, leave=True)

    
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
                lam=gae_lambda
            )
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



        N = len(meta_episodes)  # total episodes collected
        indices = np.arange(N)

        for opt_epoch in range(ppo_opt_epochs):
            # Shuffle episodes (not timesteps)
            np.random.shuffle(meta_episodes)
            
            if comm.Get_rank() == ROOT_RANK:
                logging.info(f"pol update {pol_iter}, opt_epoch: {opt_epoch}...")
                print(f"\npol update {pol_iter}, opt_epoch: {opt_epoch}...\n")
                if show_pbar:
                    # total number of minibatches
                    total_mb = (N // meta_ep_per_learner_batch)
                    weight_pbar = tqdm(total=total_mb, desc="Updating Weights")
            
            # Process episodes in mini-batches
            for start_idx in range(0, len(meta_episodes), meta_ep_per_learner_batch):
                end_idx = start_idx + meta_ep_per_learner_batch
                mb_episodes = meta_episodes[start_idx:end_idx]
                
                # Compute losses on this batch of episodes
                losses = compute_losses(
                    meta_episodes=mb_episodes,
                    policy_net=policy_net,
                    value_net=value_net,
                    clip_param=ppo_clip_param,
                    ent_coef=ppo_ent_coef
                )

                # Backprop: Policy
                policy_optimizer.zero_grad()
                losses['policy_loss'].backward()
                sync_grads(model=policy_net, comm=comm)
                # tc.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.8)
                policy_optimizer.step()
                if policy_scheduler:
                    # Use mean reward as metric for policy scheduler
                    policy_scheduler.step(np.mean(meta_ep_returns))

                # Backprop: Value
                value_optimizer.zero_grad()
                losses['value_loss'].backward()
                sync_grads(model=value_net, comm=comm)
                # tc.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=0.8)
                value_optimizer.step()
                if value_scheduler:
                    # Use value loss as metric for value scheduler
                    value_scheduler.step(losses['value_loss'].item())

                if comm.Get_rank() == ROOT_RANK and show_pbar:
                    weight_pbar.update(1)
                global_update_step += 1
                if visualize_weights_flag and comm.Get_rank() == ROOT_RANK:
                    visualize_weights(policy_net, value_net, out_dir=log_directory, update_step=global_update_step)
                

            if comm.Get_rank() == ROOT_RANK and show_pbar:
                weight_pbar.close()

            # Logging final losses from last minibatch
            global_losses = {}
            for name in losses:
                loss_sum = comm.allreduce(losses[name], op=MPI.SUM)
                loss_avg = loss_sum / comm.Get_size()
                global_losses[name] = loss_avg


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
            logging.info(f"mean episode length: {total_timesteps/episode_count:.2f}")
            print("-" * 80)
            policy_checkpoint_fn(pol_iter + 1)
            value_checkpoint_fn(pol_iter + 1)
            
        log2csv(checkpoint_dir)
