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
        
        # Optional: Context reconstruction (if we want to predict actual disturbance values)
        self.context_decoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Predict [wind_power, gravity]
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
            
        # Predict actual disturbance values (optional)
        disturbance_pred = self.context_decoder(context)
        
        return context, context_mean, context_log_std, disturbance_pred

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
        
        # Initialize networks
        self.context_encoder = ContextEncoder(
            input_dim=self.state_dim + self.action_dim + 1,  # state + action + reward
            hidden_dim=hidden_dim,
            context_dim=context_dim
        )
        
        self.policy = ContextAwarePolicy(
            state_dim=self.state_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            action_dim=self.action_dim
        )
        
        # Combine parameters for single optimizer
        self.optimizer = torch.optim.Adam(
            list(self.context_encoder.parameters()) + 
            list(self.policy.parameters()),
            lr=lr
        )
        
        # Training parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.context_kl_weight = context_kl_weight
        
        # Initialize training state
        self.total_episodes = 0
        self.training_rewards = []
        self.eval_rewards = []
        self.best_eval_reward = float('-inf')
        self.checkpoint_files = []
        
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
        state = self.env.reset()[0]
        
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
                context, context_mean, context_log_std, disturbance_pred = self.context_encoder(trajectory_tensor)
                context = context.squeeze(0)  # Remove batch dimension
        
        return states.tolist(), actions.tolist(), rewards.tolist(), log_probs.tolist(), dones.tolist(), values.tolist(), episode_reward, context, trajectory_tensor
    
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
                contexts, context_means, context_log_stds, disturbance_preds = self.context_encoder(
                    padded_trajectories
                )
                
                # Compute KL divergence loss for variational context encoder
                kl_loss = -0.5 * torch.sum(
                    1 + 2*context_log_stds - context_means.pow(2) - torch.exp(2*context_log_stds),
                    dim=-1
                ).mean()
                
                # Process states/actions/returns
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
                
                # Expand context for each timestep
                expanded_contexts = contexts.unsqueeze(0).expand(max_len, -1, -1)
                
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
                
                # Total loss
                loss = actor_loss + 0.5 * critic_loss + self.context_kl_weight * kl_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Synchronize gradients across workers
                for param in self.context_encoder.parameters():
                    if param.grad is not None:
                        grad_numpy = param.grad.data.numpy()
                        buf = np.zeros_like(grad_numpy)
                        MPI.COMM_WORLD.Allreduce([grad_numpy, MPI.FLOAT], [buf, MPI.FLOAT], op=MPI.SUM)
                        param.grad.data = torch.from_numpy(buf / self.size)
                
                for param in self.policy.parameters():
                    if param.grad is not None:
                        grad_numpy = param.grad.data.numpy()
                        buf = np.zeros_like(grad_numpy)
                        MPI.COMM_WORLD.Allreduce([grad_numpy, MPI.FLOAT], [buf, MPI.FLOAT], op=MPI.SUM)
                        param.grad.data = torch.from_numpy(buf / self.size)
                
                self.optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_context_loss += kl_loss.item()
                num_updates += 1
        
        # Average losses
        avg_actor_loss = total_actor_loss / num_updates if num_updates > 0 else 0
        avg_critic_loss = total_critic_loss / num_updates if num_updates > 0 else 0
        avg_context_loss = total_context_loss / num_updates if num_updates > 0 else 0
        
        return avg_actor_loss, avg_critic_loss, avg_context_loss
    
    def train_episode(self, num_episodes_per_update=4):
        total_reward = 0
        total_length = 0
        
        all_states = []
        all_actions = []
        all_log_probs = []
        all_returns = []
        all_contexts = []
        all_trajectories = []
        
        context = None  # Initial context estimate
        
        for _ in range(num_episodes_per_update):
            states, actions, rewards, log_probs, dones, values, episode_reward, new_context, trajectory = self.collect_episode(context)
            
            returns = self.compute_returns(rewards, dones, values)
            
            all_states.append(states)
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_returns.append(returns.tolist())
            all_contexts.append(new_context)
            all_trajectories.append(trajectory)
            
            total_reward += episode_reward
            total_length += len(states)
            
            # Update context estimate for next episode
            context = new_context
        
        actor_loss, critic_loss, context_loss = self.update_policy(
            all_states, all_actions, all_log_probs, all_returns,
            all_contexts, all_trajectories
        )
        
        avg_reward = total_reward / num_episodes_per_update
        avg_length = total_length / num_episodes_per_update
        
        return avg_reward, avg_length, actor_loss, critic_loss, context_loss
    
    def evaluate(self, num_episodes=10):
        if self.rank != 0:  # Only evaluate on rank 0
            return 0.0
            
        eval_rewards = []
        disturbance_losses = []
        context = None  # Initial context estimate
        
        for episode in range(num_episodes):
            state = self.env.reset()[0]
            episode_reward = 0
            disturbance_loss = 0
            
            # Get actual disturbances from environment
            actual_disturbances = self.env.get_current_disturbance()
            print(f"\nEpisode {episode + 1} Disturbances:")
            print(f"  Actual: gravity={actual_disturbances['gravity']:.3f}, "
                  f"wind={actual_disturbances['wind_power']:.3f}, "
                  f"turbulence={actual_disturbances['turbulence_power']:.3f}")
            
            # Pre-allocate trajectory array for max steps
            max_steps = 1000
            trajectory = np.zeros((max_steps, self.state_dim + self.action_dim + 1), dtype=np.float32)
            episode_length = 0
            done = False
            
            for step in range(max_steps):  # Use step counter to ensure we don't exceed max_steps
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
                        context, _, _, disturbance_pred = self.context_encoder(traj_tensor)
                        context = context.squeeze(0)
                        
                        # Print predicted disturbances every 100 steps
                        if step % 100 == 0:
                            # Predicted disturbances: first value is wind_power, second is gravity
                            pred_wind, pred_gravity = disturbance_pred.squeeze().numpy()
                            # Scale predictions to match actual ranges
                            pred_wind = pred_wind * 7.5 + 7.5  # Scale to [0, 15]
                            pred_gravity = pred_gravity * 1.0 - 10.0  # Scale to [-11, -9]
                            print(f"  Step {step:4d} Predicted: gravity={pred_gravity:.3f}, wind={pred_wind:.3f}")
                
                if done or step == max_steps - 1:  # Break if done or reached max steps
                    # Print final predictions
                    pred_wind, pred_gravity = disturbance_pred.squeeze().numpy()
                    pred_wind = pred_wind * 7.5 + 7.5  # Scale to [0, 15]
                    pred_gravity = pred_gravity * 1.0 - 10.0  # Scale to [-11, -9]
                    disturbance_loss = np.sqrt((pred_wind - actual_disturbances['wind_power'])**2 + (pred_gravity - actual_disturbances['gravity'])**2)
                    print(f"  Final Predicted: gravity={pred_gravity:.3f}, wind={pred_wind:.3f}")
                    print(f"  Disturbance Loss: {disturbance_loss:.2f}")
                    print(f"  Episode Length: {episode_length}, Reward: {episode_reward:.2f}")
                    break
            
            # Save evaluation metrics
            eval_rewards.append(episode_reward)
            disturbance_losses.append(disturbance_loss)
            
        mean_reward = np.mean(eval_rewards)
        print(f"\nEvaluation Summary:")
        print(f"Reward over {num_episodes} episodes: {mean_reward:.2f} +- {np.std(eval_rewards):.2f}")
        print(f"Disturbance Loss: {np.mean(disturbance_losses):.2f}")
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
            'optimizer_state_dict': self.optimizer.state_dict(),
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
        instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        instance.total_episodes = checkpoint.get('total_episodes', 0)
        instance.training_rewards = checkpoint.get('training_rewards', [])
        instance.eval_rewards = checkpoint.get('eval_rewards', [])
        instance.best_eval_reward = checkpoint.get('best_eval_reward', float('-inf'))
        
        if 'run_dir' in checkpoint:
            instance.run_dir = checkpoint['run_dir']
            instance.checkpoint_dir = checkpoint['checkpoint_dir']
            instance.log_dir = checkpoint['log_dir']
        
        return instance 