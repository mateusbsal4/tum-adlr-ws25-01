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

class ContextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, context_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=False)
        
        # Variational encoder (outputs mean and log_std of context distribution)
        self.context_mean = nn.Linear(hidden_dim, context_dim)
        self.context_log_std = nn.Linear(hidden_dim, context_dim)
        
        # Wind power predictor (single value output)
        self.wind_predictor = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Predict wind_power directly
            nn.Sigmoid()  # Scale output to [0,1], will be rescaled to [0,15]
        )
        
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
            
        # Predict wind power (scaled to [0,15])
        wind_power = self.wind_predictor(context) * 15.0
        
        return context, context_mean, context_log_std, wind_power

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
        )
        
        self.policy = ContextAwarePolicy(
            state_dim=self.state_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            action_dim=self.action_dim
        )
        
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
        state = torch.FloatTensor(state).unsqueeze(0)
        context = context.unsqueeze(0) if context is not None else torch.zeros(1, self.context_dim)
        
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
            context = torch.zeros(self.context_dim)
        
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
            # Convert trajectory to tensor efficiently
            trajectory_tensor = torch.from_numpy(trajectory).unsqueeze(1)  # [seq_len, 1, dim]
            with torch.no_grad():
                context, context_mean, context_log_std, wind_power = self.context_encoder(trajectory_tensor)
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
        returns = torch.from_numpy(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def update_policy(self, all_states, all_actions, all_log_probs, all_returns, all_contexts, all_trajectories):
        # Convert all inputs to tensors efficiently
        episodes_states = [torch.from_numpy(np.array(states, dtype=np.float32)) for states in all_states]
        episodes_actions = [torch.from_numpy(np.array(actions, dtype=np.int64)) for actions in all_actions]
        episodes_old_log_probs = [torch.from_numpy(np.array(log_probs, dtype=np.float32)) for log_probs in all_log_probs]
        episodes_returns = [torch.from_numpy(np.array(returns, dtype=np.float32)) for returns in all_returns]
        
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
                padded_trajectories = torch.zeros(max_traj_len, actual_batch_size, batch_trajectories[0].size(-1))
                trajectory_masks = torch.zeros(max_traj_len, actual_batch_size)
                
                for i, traj in enumerate(batch_trajectories):
                    traj_len = traj.size(0)
                    padded_trajectories[:traj_len, i] = traj.squeeze(1)
                    trajectory_masks[:traj_len, i] = 1
                
                # Encode contexts
                contexts, context_means, context_log_stds, wind_powers = self.context_encoder(
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
                self.context_optimizer.step()
                
                # Process states/actions/returns with detached context
                max_len = max(episodes_states[idx].size(0) for idx in batch_indices)
                
                batch_states = torch.zeros(max_len, actual_batch_size, self.state_dim)
                batch_actions = torch.zeros(max_len, actual_batch_size, dtype=torch.long)
                batch_old_log_probs = torch.zeros(max_len, actual_batch_size)
                batch_returns = torch.zeros(max_len, actual_batch_size)
                batch_masks = torch.zeros(max_len, actual_batch_size)
                
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
                self.policy_optimizer.step()
                
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
    
    def train_episode(self, num_episodes_per_update=1):
        # Update curriculum before episode
        self.update_curriculum()
        
        # Single episode collection
        states = []
        actions = []
        log_probs = []
        returns = []
        contexts = []
        trajectories = []
        
        context = None  # Initial context estimate
        
        # Environment will sample new wind power and target position on reset
        state = self.env.reset()[0]
        if self.rank == 0:  # Only print from rank 0
            print(f"\nNew wind power: {self.env.wind_power:.3f}")
            print(f"New target position: {self.env.target_x:.3f}")
        
        # Collect single episode
        states_ep, actions_ep, rewards_ep, log_probs_ep, dones_ep, values_ep, episode_reward, new_context, trajectory = self.collect_episode(context)
        
        # Compute returns for this episode
        returns_ep = self.compute_returns(rewards_ep, dones_ep, values_ep)
        
        # Store episode data
        states = [states_ep]
        actions = [actions_ep]
        log_probs = [log_probs_ep]
        returns = [returns_ep.tolist()]
        contexts = [new_context]
        trajectories = [trajectory]
        
        # Update policy after single episode
        actor_loss, critic_loss, context_loss = self.update_policy(
            states, actions, log_probs, returns,
            contexts, trajectories
        )
        
        return episode_reward, len(states_ep), actor_loss, critic_loss, context_loss
    
    def evaluate(self, num_episodes=10):
        if self.rank != 0:  # Only evaluate on rank 0
            return 0.0
            
        eval_rewards = []
        wind_prediction_errors = []
        target_errors = []
        context = None  # Initial context estimate
        
        for episode in range(num_episodes):
            # Environment will sample new wind power and target position on reset
            state = self.env.reset()[0]
            actual_wind = self.env.wind_power
            actual_target = self.env.target_x
            
            print(f"\nEvaluation Episode {episode + 1}")
            print(f"  Actual wind power: {actual_wind:.3f}")
            print(f"  Actual target position: {actual_target:.3f}")
            
            # Pre-allocate trajectory array for max steps
            max_steps = 1000
            trajectory = np.zeros((max_steps, self.state_dim + self.action_dim + 1), dtype=np.float32)
            episode_length = 0
            done = False
            episode_reward = 0
            
            for step in range(max_steps):
                # Get action using current context estimate
                action, _, _ = self.get_action(state, context)
                next_state, reward, done, truncated, _ = self.env.step(action)
                
                # Store transition in trajectory array
                action_one_hot = np.zeros(self.action_dim, dtype=np.float32)
                action_one_hot[action] = 1
                trajectory[episode_length] = np.concatenate([state, action_one_hot, [reward]])
                
                episode_reward += reward
                state = next_state
                episode_length += 1
                done = done or truncated
                
                # Update context estimate using trajectory so far
                if episode_length > 0:
                    # Trim trajectory to current length and convert to tensor
                    traj_tensor = torch.from_numpy(trajectory[:episode_length]).unsqueeze(1)
                    with torch.no_grad():
                        context, _, _, wind_pred = self.context_encoder(traj_tensor)
                        context = context.squeeze(0)
                        
                        # Print predictions every 100 steps
                        if step % 100 == 0:
                            print(f"  Step {step:4d} Predicted wind: {wind_pred.item():.3f}")
                
                if done or step == max_steps - 1:
                    wind_error = abs(wind_pred.item() - actual_wind)
                    print(f"  Final Predicted wind: {wind_pred.item():.3f}")
                    print(f"  Wind Prediction Error: {wind_error:.3f}")
                    print(f"  Episode Length: {episode_length}, Reward: {episode_reward:.2f}")
                    wind_prediction_errors.append(wind_error)
                    break
            
            eval_rewards.append(episode_reward)
        
        mean_reward = np.mean(eval_rewards)
        mean_wind_error = np.mean(wind_prediction_errors)
        print(f"\nEvaluation Summary:")
        print(f"Average Reward: {mean_reward:.2f} ± {np.std(eval_rewards):.2f}")
        print(f"Average Wind Prediction Error: {mean_wind_error:.3f} ± {np.std(wind_prediction_errors):.3f}")
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