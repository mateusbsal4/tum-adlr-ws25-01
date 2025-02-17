import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, Categorical
import gymnasium as gym
from collections import deque
import os
from datetime import datetime
from mpi4py import MPI
from custom_lunar_lander import CustomLunarLander
import matplotlib.pyplot as plt

class ContextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, context_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=False)
        
        # Variational encoder (outputs mean and log_std of context distribution)
        self.context_mean = nn.Linear(hidden_dim, context_dim)
        self.context_log_std = nn.Linear(hidden_dim, context_dim)
        
        # Disturbance predictor network
        self.disturbance_predictor = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Predict [wind_power, target_x]
        )
        
        # Output activation layers
        self.wind_activation = nn.Sigmoid()  # Scale wind to [0,1]
        self.pos_activation = nn.Sigmoid()   # Scale position to [0,1]
        
    def forward(self, x, hidden=None):
        # x shape: [seq_len, batch_size, input_dim]
        if hidden is None:
            batch_size = x.size(1) 
            hidden = torch.zeros(1, batch_size, self.gru.hidden_size, device=x.device)
        
        # Encode trajectory
        _, hidden = self.gru(x, hidden)
        hidden = hidden.squeeze(0)  # [batch_size, hidden_dim]
        
        # Get distribution parameters
        context_mean = self.context_mean(hidden)
        context_log_std = self.context_log_std(hidden)
        context_std = torch.exp(context_log_std)
        
        # Sample context using reparameterization trick
        if self.training:
            eps = torch.randn_like(context_std)
            context = context_mean + context_std * eps
        else:
            context = context_mean
            
        # Predict disturbances
        disturbance_pred = self.disturbance_predictor(context)
        wind_power = self.wind_activation(disturbance_pred[..., 0]) * 15.0  # Scale to [0,15]
        target_pos = self.pos_activation(disturbance_pred[..., 1]) * 0.6 + 0.2  # Scale to [0.2,0.8]
        
        return context, context_mean, context_log_std, wind_power, target_pos

class ContextAwarePolicy(nn.Module):
    def __init__(self, state_dim, context_dim, hidden_dim, action_dim):
        super().__init__()
        self.state_context_net = nn.Sequential(
            nn.Linear(state_dim + context_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, context):
        # Concatenate state and context
        x = torch.cat([state, context], dim=-1)
        features = self.state_context_net(x)
        
        # Get action logits and value
        action_logits = self.actor(features)
        value = self.critic(features)
        
        return action_logits, value

class ContextAwarePPO:
    def __init__(self, env_name="CustomLunarLander-v2", hidden_dim=128, context_dim=8,
                 lr=3e-4, gamma=0.99, epsilon=0.2, epochs=10, batch_size=64,
                 context_kl_weight=0.1, render_mode=None, run_name=None):
        # Initialize MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.rank == 0:
            print(f"Using device: {self.device}")
        
        # Create directories
        if self.rank == 0:
            os.makedirs('models', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_name = run_name or f'run_{timestamp}'
            self.run_dir = os.path.join('models', run_name)
            os.makedirs(self.run_dir, exist_ok=True)
            self.checkpoint_dir = os.path.join(self.run_dir, 'checkpoints')
            self.log_dir = os.path.join(self.run_dir, 'logs')
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)
        
        # Broadcast run directories to all processes
        if self.size > 1:
            if self.rank == 0:
                dirs = [self.run_dir, self.checkpoint_dir, self.log_dir]
            else:
                dirs = [None, None, None]
            dirs = self.comm.bcast(dirs, root=0)
            if self.rank != 0:
                self.run_dir, self.checkpoint_dir, self.log_dir = dirs
        
        # Create environment
        if env_name == "CustomLunarLander-v2":
            self.env = CustomLunarLander(render_mode=render_mode if self.rank == 0 else None)
        else:
            self.env = gym.make(env_name, render_mode=render_mode if self.rank == 0 else None)
            
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.env_name = env_name
        self.render_mode = render_mode
        
        # Initialize networks with better architectures
        self.context_encoder = ContextEncoder(
            input_dim=self.state_dim + self.action_dim + 1,
            hidden_dim=hidden_dim,
            context_dim=context_dim
        ).to(self.device)
        
        self.policy = ContextAwarePolicy(
            state_dim=self.state_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            action_dim=self.action_dim
        ).to(self.device)
        
        # Separate optimizers for policy and context encoder
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=lr
        )
        self.context_optimizer = torch.optim.Adam(
            self.context_encoder.parameters(),
            lr=lr * 0.5  # Slower learning rate for context encoder
        )
        
        # Learning rate schedulers
        self.policy_scheduler = torch.optim.lr_scheduler.StepLR(
            self.policy_optimizer, 
            step_size=1000,  # Adjust every 1000 episodes
            gamma=0.95  # Reduce learning rate by 5%
        )
        self.context_scheduler = torch.optim.lr_scheduler.StepLR(
            self.context_optimizer,
            step_size=1000,
            gamma=0.95
        )
        
        # Adaptive context KL weight
        self.context_kl_weight = context_kl_weight
        self.min_kl_weight = 0.01
        self.max_kl_weight = 0.5
        self.target_kl = 0.02
        
        # Training parameters with curriculum
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Initialize training state
        self.total_episodes = 0
        self.training_rewards = []
        self.eval_rewards = []
        self.best_eval_reward = float('-inf')
        self.checkpoint_files = []
        
        # Curriculum learning parameters
        self.wind_range = (0.0, 5.0)  # Start with smaller wind range
        self.target_range = (0.4, 0.6)  # Start with smaller target range
        self.curriculum_step = 0
        
    def update_curriculum(self):
        """Update curriculum based on performance"""
        if len(self.eval_rewards) > 0 and len(self.training_rewards) > 100:
            avg_reward = np.mean(self.training_rewards[-100:])
            if avg_reward > -200:  # If performance is good enough
                self.curriculum_step += 1
                if self.curriculum_step == 1:
                    self.wind_range = (0.0, 10.0)  # Increase wind range
                    self.target_range = (0.3, 0.7)  # Increase target range
                elif self.curriculum_step == 2:
                    self.wind_range = (0.0, 15.0)  # Full wind range
                    self.target_range = (0.2, 0.8)  # Full target range
                
                if self.rank == 0:
                    print(f"\nUpdating curriculum - Step {self.curriculum_step}")
                    print(f"New wind range: {self.wind_range}")
                    print(f"New target range: {self.target_range}")
    
    def get_action(self, state, context):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        context = context.unsqueeze(0).to(self.device) if context is not None else torch.zeros(1, self.context_dim).to(self.device)
        
        with torch.no_grad():
            action_logits, value = self.policy(state, context)
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item(), value.item()
    
    def collect_episode(self, context=None):
        # Reset without sampling new wind speed (it's already set for this update)
        state = self.env.reset(options={'sample_new_wind': False})[0]
        
        # Pre-allocate numpy arrays for max episode length
        max_steps = 1000
        states = np.zeros((max_steps, self.state_dim), dtype=np.float32)
        actions = np.zeros(max_steps, dtype=np.int64)
        rewards = np.zeros(max_steps, dtype=np.float32)
        log_probs = np.zeros(max_steps, dtype=np.float32)
        dones = np.zeros(max_steps, dtype=np.bool_)
        values = np.zeros(max_steps, dtype=np.float32)
        
        # Pre-allocate trajectory array (state + one-hot action + reward)
        trajectory = np.zeros((max_steps, self.state_dim + self.action_dim + 1), dtype=np.float32)
        
        if context is None:
            context = torch.zeros(self.context_dim).to(self.device)
        
        episode_length = 0
        episode_reward = 0
        
        for step in range(max_steps):
            if self.render_mode == "human":
                self.env.render()
            
            # Get action using current context estimate
            action, log_prob, value = self.get_action(state, context)
            next_state, reward, done, truncated, _ = self.env.step(action)
            
            # Store transition in pre-allocated arrays
            states[step] = state
            actions[step] = action
            rewards[step] = reward
            log_probs[step] = log_prob
            dones[step] = done or truncated
            values[step] = value
            
            # Store trajectory for context encoding
            # One-hot encode action
            action_one_hot = np.zeros(self.action_dim, dtype=np.float32)
            action_one_hot[action] = 1
            trajectory[step] = np.concatenate([state, action_one_hot, [reward]])
            
            episode_reward += reward
            state = next_state
            episode_length += 1
            
            if done or truncated:
                break
        
        # Trim arrays to actual episode length
        states = states[:episode_length]
        actions = actions[:episode_length]
        rewards = rewards[:episode_length]
        log_probs = log_probs[:episode_length]
        dones = dones[:episode_length]
        values = values[:episode_length]
        trajectory = trajectory[:episode_length]
        
        # Encode trajectory to get context
        if episode_length > 0:
            trajectory_tensor = torch.from_numpy(trajectory).unsqueeze(1).to(self.device)  # [seq_len, 1, dim]
            with torch.no_grad():
                context, context_mean, context_log_std, wind_power, target_pos = self.context_encoder(trajectory_tensor)
                context = context.squeeze(0)  # Remove batch dimension
        
        return states, actions, rewards, log_probs, dones, values, episode_reward, context, trajectory_tensor
    
    def compute_returns(self, rewards, dones, values):
        # Convert inputs to numpy arrays if they aren't already
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        values = np.array(values, dtype=np.float32)
        
        # Pre-allocate returns array
        returns = np.zeros_like(rewards)
        running_return = 0
        
        # Compute returns in reverse order
        for t in range(len(rewards) - 1, -1, -1):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns[t] = running_return
            
        # Convert to tensor and normalize
        returns = torch.from_numpy(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def sync_gradients(self):
        """Synchronize gradients across all MPI processes"""
        for model in [self.context_encoder, self.policy]:
            for param in model.parameters():
                if param.grad is not None:
                    # All-reduce gradients
                    param_grad = param.grad.data.cpu().numpy()
                    self.comm.Allreduce(MPI.IN_PLACE, param_grad, op=MPI.SUM)
                    param_grad = param_grad / self.size
                    param.grad.data = torch.from_numpy(param_grad).to(self.device)

    def sync_networks(self):
        """Synchronize network parameters across all MPI processes"""
        for model in [self.context_encoder, self.policy]:
            for param in model.parameters():
                param_data = param.data.cpu().numpy()
                self.comm.Bcast(param_data, root=0)
                param.data = torch.from_numpy(param_data).to(self.device)

    def update_policy(self, all_states, all_actions, all_log_probs, all_returns, all_contexts, all_trajectories):
        # Convert all inputs to tensors efficiently
        episodes_states = [torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device) for states in all_states]
        episodes_actions = [torch.from_numpy(np.array(actions, dtype=np.int64)).to(self.device) for actions in all_actions]
        episodes_old_log_probs = [torch.from_numpy(np.array(log_probs, dtype=np.float32)).to(self.device) for log_probs in all_log_probs]
        episodes_returns = [torch.from_numpy(np.array(returns, dtype=np.float32)).to(self.device) for returns in all_returns]
        
        num_episodes = len(episodes_states)
        total_actor_loss = 0
        total_critic_loss = 0
        total_context_loss = 0
        num_updates = 0
        
        for _ in range(self.epochs):
            episode_indices = np.random.permutation(num_episodes)
            
            for start_idx in range(0, num_episodes, self.batch_size):
                batch_indices = episode_indices[start_idx:start_idx + self.batch_size]
                actual_batch_size = len(batch_indices)
                
                # Process trajectories for context encoding
                batch_trajectories = [all_trajectories[i] for i in batch_indices]
                max_traj_len = max(traj.size(0) for traj in batch_trajectories)
                
                # Pad trajectories
                padded_trajectories = torch.zeros(max_traj_len, actual_batch_size, batch_trajectories[0].size(-1)).to(self.device)
                trajectory_masks = torch.zeros(max_traj_len, actual_batch_size).to(self.device)
                
                for i, traj in enumerate(batch_trajectories):
                    traj_len = traj.size(0)
                    padded_trajectories[:traj_len, i] = traj.squeeze(1)
                    trajectory_masks[:traj_len, i] = 1
                
                # Encode contexts
                contexts, context_means, context_log_stds, wind_powers, target_positions = self.context_encoder(
                    padded_trajectories
                )
                
                # Compute KL divergence loss for variational context encoder
                kl_loss = -0.5 * torch.sum(
                    1 + 2*context_log_stds - context_means.pow(2) - torch.exp(2*context_log_stds),
                    dim=-1
                ).mean()
                
                # Context update first
                context_loss = self.context_kl_weight * kl_loss
                self.context_optimizer.zero_grad()
                context_loss.backward(retain_graph=True)
                
                # Synchronize context encoder gradients across processes
                self.sync_gradients()
                
                self.context_optimizer.step()
                
                # Process states/actions/returns with detached context
                max_len = max(episodes_states[idx].size(0) for idx in batch_indices)
                
                batch_states = torch.zeros(max_len, actual_batch_size, self.state_dim).to(self.device)
                batch_actions = torch.zeros(max_len, actual_batch_size, dtype=torch.long).to(self.device)
                batch_old_log_probs = torch.zeros(max_len, actual_batch_size).to(self.device)
                batch_returns = torch.zeros(max_len, actual_batch_size).to(self.device)
                batch_masks = torch.zeros(max_len, actual_batch_size).to(self.device)
                
                for batch_idx, episode_idx in enumerate(batch_indices):
                    episode_len = episodes_states[episode_idx].size(0)
                    batch_states[:episode_len, batch_idx] = episodes_states[episode_idx]
                    batch_actions[:episode_len, batch_idx] = episodes_actions[episode_idx]
                    batch_old_log_probs[:episode_len, batch_idx] = episodes_old_log_probs[episode_idx]
                    batch_returns[:episode_len, batch_idx] = episodes_returns[episode_idx]
                    batch_masks[:episode_len, batch_idx] = 1
                
                # Create a detached copy of contexts for policy update
                contexts_for_policy = contexts.detach().clone()
                expanded_contexts = contexts_for_policy.unsqueeze(0).expand(max_len, -1, -1)
                
                # Get policy predictions
                action_logits, values = self.policy(batch_states, expanded_contexts)
                
                # Compute policy loss
                dist = Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(batch_actions)
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_returns
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_returns
                
                actor_loss = -torch.min(surr1, surr2) * batch_masks
                critic_loss = (values.squeeze(-1) - batch_returns).pow(2) * batch_masks
                
                actor_loss = actor_loss.sum() / batch_masks.sum()
                critic_loss = critic_loss.sum() / batch_masks.sum()
                
                # Policy update with detached context
                policy_loss = actor_loss + 0.5 * critic_loss
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                
                # Synchronize policy gradients across processes
                self.sync_gradients()
                
                self.policy_optimizer.step()
                
                # Synchronize networks after updates
                self.sync_networks()
                
                # Adjust KL weight based on KL divergence
                if kl_loss < self.target_kl:
                    self.context_kl_weight = max(self.min_kl_weight, self.context_kl_weight * 0.95)
                else:
                    self.context_kl_weight = min(self.max_kl_weight, self.context_kl_weight * 1.05)
                
                # Update learning rates
                self.policy_scheduler.step()
                self.context_scheduler.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_context_loss += kl_loss.item()
                num_updates += 1
        
        # Average losses
        avg_actor_loss = total_actor_loss / num_updates if num_updates > 0 else 0
        avg_critic_loss = total_critic_loss / num_updates if num_updates > 0 else 0
        avg_context_loss = total_context_loss / num_updates if num_updates > 0 else 0
        
        return avg_actor_loss, avg_critic_loss, avg_context_loss
    
    def collect_parallel_episodes(self, num_episodes_per_update=1):
        """Collect episodes in parallel across MPI processes"""
        # Each process collects its share of episodes
        episodes_per_process = num_episodes_per_update // self.size
        if self.rank < num_episodes_per_update % self.size:
            episodes_per_process += 1
            
        # Local storage for this process
        local_states = []
        local_actions = []
        local_log_probs = []
        local_returns = []
        local_contexts = []
        local_trajectories = []
        local_rewards = []
        
        # Collect episodes for this process
        for _ in range(episodes_per_process):
            states_ep, actions_ep, rewards_ep, log_probs_ep, dones_ep, values_ep, episode_reward, new_context, trajectory = self.collect_episode(None)
            returns_ep = self.compute_returns(rewards_ep, dones_ep, values_ep)
            
            local_states.append(states_ep)
            local_actions.append(actions_ep)
            local_log_probs.append(log_probs_ep)
            local_returns.append(returns_ep.tolist())
            local_contexts.append(new_context)
            local_trajectories.append(trajectory)
            local_rewards.append(episode_reward)
        
        # Gather data from all processes
        all_states = self.comm.gather(local_states, root=0)
        all_actions = self.comm.gather(local_actions, root=0)
        all_log_probs = self.comm.gather(local_log_probs, root=0)
        all_returns = self.comm.gather(local_returns, root=0)
        all_contexts = self.comm.gather(local_contexts, root=0)
        all_trajectories = self.comm.gather(local_trajectories, root=0)
        all_rewards = self.comm.gather(local_rewards, root=0)
        
        # Process 0 combines the data
        if self.rank == 0:
            all_states = [ep for proc_states in all_states for ep in proc_states]
            all_actions = [ep for proc_actions in all_actions for ep in proc_actions]
            all_log_probs = [ep for proc_log_probs in all_log_probs for ep in proc_log_probs]
            all_returns = [ep for proc_returns in all_returns for ep in proc_returns]
            all_contexts = [ep for proc_contexts in all_contexts for ep in proc_contexts]
            all_trajectories = [ep for proc_trajectories in all_trajectories for ep in proc_trajectories]
            all_rewards = [r for proc_rewards in all_rewards for r in proc_rewards]
            
            avg_reward = np.mean(all_rewards)
            avg_length = np.mean([len(states) for states in all_states])
        else:
            all_states = None
            all_actions = None
            all_log_probs = None
            all_returns = None
            all_contexts = None
            all_trajectories = None
            avg_reward = None
            avg_length = None
            
        # Broadcast the combined data to all processes
        all_states = self.comm.bcast(all_states, root=0)
        all_actions = self.comm.bcast(all_actions, root=0)
        all_log_probs = self.comm.bcast(all_log_probs, root=0)
        all_returns = self.comm.bcast(all_returns, root=0)
        all_contexts = self.comm.bcast(all_contexts, root=0)
        all_trajectories = self.comm.bcast(all_trajectories, root=0)
        avg_reward = self.comm.bcast(avg_reward, root=0)
        avg_length = self.comm.bcast(avg_length, root=0)
        
        return all_states, all_actions, all_log_probs, all_returns, all_contexts, all_trajectories, avg_reward, avg_length
    
    def train_episode(self, num_episodes_per_update=1):
        # Update curriculum before episode
        self.update_curriculum()
        
        # Collect episodes in parallel
        all_states, all_actions, all_log_probs, all_returns, all_contexts, all_trajectories, avg_reward, avg_length = self.collect_parallel_episodes(num_episodes_per_update)
        
        # Update policy with collected data
        actor_loss, critic_loss, context_loss = self.update_policy(
            all_states, all_actions, all_log_probs, all_returns,
            all_contexts, all_trajectories
        )
        
        return avg_reward, avg_length, actor_loss, critic_loss, context_loss
    
    def evaluate(self, num_episodes=10):
        if self.rank != 0:  # Only evaluate on rank 0
            return 0.0
            
        eval_rewards = []
        wind_prediction_errors = []
        position_prediction_errors = []
        context = None
        
        for episode in range(num_episodes):
            state = self.env.reset()[0]
            actual_wind = self.env.wind_power
            actual_target = self.env.target_x
            
            print(f"\nEvaluation Episode {episode + 1}")
            print(f"  Actual wind power: {actual_wind:.3f}")
            print(f"  Actual target position: {actual_target:.3f}")
            
            max_steps = 1000
            trajectory = np.zeros((max_steps, self.state_dim + self.action_dim + 1), dtype=np.float32)
            episode_length = 0
            done = False
            episode_reward = 0
            
            for step in range(max_steps):
                action, _, _ = self.get_action(state, context)
                next_state, reward, done, truncated, _ = self.env.step(action)
                
                action_one_hot = np.zeros(self.action_dim, dtype=np.float32)
                action_one_hot[action] = 1
                trajectory[episode_length] = np.concatenate([state, action_one_hot, [reward]])
                
                episode_reward += reward
                state = next_state
                episode_length += 1
                done = done or truncated
                
                # Update context estimate using trajectory so far
                if episode_length > 0:
                    traj_tensor = torch.from_numpy(trajectory[:episode_length]).unsqueeze(1)
                    with torch.no_grad():
                        context, _, _, wind_pred, pos_pred = self.context_encoder(traj_tensor)
                        context = context.squeeze(0)
                        
                        # # Print predictions every 100 steps
                        # if step % 100 == 0:
                        #     print(f"  Step {step:4d}:")
                        #     print(f"    Predicted wind: {wind_pred.item():.3f}")
                        #     print(f"    Predicted position: {pos_pred.item():.3f}")
                
                if done or step == max_steps - 1:
                    wind_error = abs(wind_pred.item() - actual_wind)
                    pos_error = abs(pos_pred.item() - actual_target)
                    print(f"\n  Final Predictions:")
                    print(f"    Wind Power - Predicted: {wind_pred.item():.3f}, Actual: {actual_wind:.3f}, Error: {wind_error:.3f}")
                    print(f"    Target Pos - Predicted: {pos_pred.item():.3f}, Actual: {actual_target:.3f}, Error: {pos_error:.3f}")
                    print(f"    Episode Length: {episode_length}, Reward: {episode_reward:.2f}")
                    wind_prediction_errors.append(wind_error)
                    position_prediction_errors.append(pos_error)
                    break
            
            eval_rewards.append(episode_reward)
        
        mean_reward = np.mean(eval_rewards)
        mean_wind_error = np.mean(wind_prediction_errors)
        mean_pos_error = np.mean(position_prediction_errors)
        
        print(f"\nEvaluation Summary:")
        print(f"Average Reward: {mean_reward:.2f} ± {np.std(eval_rewards):.2f}")
        print(f"Average Wind Prediction Error: {mean_wind_error:.3f} ± {np.std(wind_prediction_errors):.3f}")
        print(f"Average Position Prediction Error: {mean_pos_error:.3f} ± {np.std(position_prediction_errors):.3f}")
        
        return mean_reward
    
    def save_model(self, suffix=''):
        if self.rank != 0:  # Only save on rank 0
            return None
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'model_step{self.total_episodes}_{timestamp}{suffix}.pt'
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        torch.save({
            'context_encoder_state_dict': self.context_encoder.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.policy_optimizer.state_dict(),
            'total_episodes': self.total_episodes,
            'training_rewards': self.training_rewards,
            'eval_rewards': self.eval_rewards,
            'best_eval_reward': self.best_eval_reward,
            'env_name': self.env_name,
            'hidden_dim': self.hidden_dim,
            'context_dim': self.context_dim,
            'run_dir': self.run_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir
        }, filepath)
        
        print(f"Model saved to {filepath}")
        return filepath
    
    @classmethod
    def load_model(cls, model_path, render_mode=None):
        checkpoint = torch.load(model_path)
        
        instance = cls(
            env_name=checkpoint['env_name'],
            hidden_dim=checkpoint['hidden_dim'],
            context_dim=checkpoint['context_dim'],
            render_mode=render_mode
        )
        
        instance.context_encoder.load_state_dict(checkpoint['context_encoder_state_dict'])
        instance.policy.load_state_dict(checkpoint['policy_state_dict'])
        instance.policy_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        instance.total_episodes = checkpoint.get('total_episodes', 0)
        instance.training_rewards = checkpoint.get('training_rewards', [])
        instance.eval_rewards = checkpoint.get('eval_rewards', [])
        instance.best_eval_reward = checkpoint.get('best_eval_reward', float('-inf'))
        
        if 'run_dir' in checkpoint:
            instance.run_dir = checkpoint['run_dir']
            instance.checkpoint_dir = checkpoint['checkpoint_dir']
            instance.log_dir = checkpoint['log_dir']
        
        return instance 

    def plot_disturbance_reconstruction(self, episode_data):
        """
        Plot the actual vs predicted wind power and target position over an episode.
        
        Args:
            episode_data: dict containing trajectories and actual values
        """
        trajectory = episode_data['trajectory']
        actual_wind = episode_data['actual_wind']
        actual_target = episode_data['actual_target']
        
        # Get predictions for each timestep
        wind_preds = []
        pos_preds = []
        
        for t in range(len(trajectory)):
            traj_tensor = torch.from_numpy(trajectory[:t+1]).unsqueeze(1)
            with torch.no_grad():
                _, _, _, wind_pred, pos_pred = self.context_encoder(traj_tensor)
                wind_preds.append(wind_pred.item())
                pos_preds.append(pos_pred.item())
        
        timesteps = np.arange(len(trajectory))
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot wind power
        ax1.plot(timesteps, [actual_wind] * len(timesteps), 'r--', label='Actual Wind')
        ax1.plot(timesteps, wind_preds, 'b-', label='Predicted Wind')
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Wind Power')
        ax1.legend()
        ax1.grid(True)
        
        # Plot target position
        ax2.plot(timesteps, [actual_target] * len(timesteps), 'r--', label='Actual Target')
        ax2.plot(timesteps, pos_preds, 'b-', label='Predicted Target')
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Target Position')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        return fig 