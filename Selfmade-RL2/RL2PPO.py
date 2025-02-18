import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque
import os
from datetime import datetime
import torch.nn.utils as nn_utils
from mpi4py import MPI
from custom_lunar_lander import CustomLunarLander
import re
import warnings

def mpi_avg_grads(model):
    """Average gradients across all MPI processes."""
    size = float(MPI.COMM_WORLD.Get_size())
    for param in model.parameters():
        if param.grad is not None:
            # Create input and output buffers
            grad_numpy = param.grad.data.numpy()
            buf = np.zeros_like(grad_numpy)
            # Perform the reduction
            MPI.COMM_WORLD.Allreduce(grad_numpy, buf, op=MPI.SUM)
            # Update the gradient
            param.grad.data = torch.from_numpy(buf / size)

def mpi_avg(x):
    """Average a float/numpy array across MPI processes."""
    if not isinstance(x, np.ndarray):
        x = np.array([x])
    buf = np.zeros_like(x)
    MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
    result = buf / MPI.COMM_WORLD.Get_size()
    # Return scalar if input was scalar
    return float(result[0]) if len(result) == 1 else result

class RL2PPO:
    def __init__(self, env_name="CustomLunarLander-v2", hidden_dim=128, lr=3e-4, gamma=0.99, epsilon=0.2, 
                 epochs=10, batch_size=64, render_mode=None, run_name=None):
        # Initialize MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Only create directories on rank 0
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
        
        # Create custom environment for each worker
        if env_name == "CustomLunarLander-v2":
            self.env = CustomLunarLander(render_mode=render_mode if self.rank == 0 else None)
        else:
            self.env = gym.make(env_name, render_mode=render_mode if self.rank == 0 else None)
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.n
        self.hidden_dim = hidden_dim
        self.env_name = env_name
        self.render_mode = render_mode
        
        self.network = GRUNetwork(self.input_dim, hidden_dim, self.output_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
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
        self.best_model_files = []

    def save_model(self, suffix='', keep_last_n=5):
        """Save the model with update steps count and optional suffix."""
        if self.rank != 0:
            return None
        
        # Calculate update steps (total_episodes / episodes_per_update)
        update_steps = self.total_episodes // 4  # assuming num_episodes_per_update=4
        
        # Determine the save directory and file list based on suffix
        if suffix == '_checkpoint':
            save_dir = self.checkpoint_dir
            file_list = self.checkpoint_files
            filename = f"{self.env_name}_step{update_steps}{suffix}.pth"
        elif suffix == '_best':
            save_dir = os.path.join(self.run_dir, 'best_models')
            os.makedirs(save_dir, exist_ok=True)
            file_list = self.best_model_files
            filename = f"{self.env_name}_step{update_steps}_reward{self.best_eval_reward:.0f}.pth"
        else:
            save_dir = self.run_dir
            file_list = None  # Don't manage other types of saves
            filename = f"{self.env_name}_step{update_steps}{suffix}.pth"
        
        filepath = os.path.join(save_dir, filename)
        
        # Save model state along with relevant parameters and training history
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hidden_dim': self.hidden_dim,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'env_name': self.env_name,
            'total_episodes': self.total_episodes,
            'update_steps': update_steps,
            'training_rewards': self.training_rewards,
            'eval_rewards': self.eval_rewards,
            'best_eval_reward': self.best_eval_reward,
            'run_dir': self.run_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir
        }, filepath)
        
        # Manage file history if specified
        if file_list is not None:
            file_list.append(filepath)
            if len(file_list) > keep_last_n:
                # Sort files by update step number for proper cleanup
                file_list.sort(key=lambda x: int(re.search(r'step(\d+)', x).group(1)))
                while len(file_list) > keep_last_n:
                    old_file = file_list.pop(0)  # Remove oldest file
                    if os.path.exists(old_file):
                        os.remove(old_file)
                        print(f"Removed old file: {old_file}")
        
        print(f"Model saved to {filepath}")
        return filepath

    @classmethod
    def load_model(cls, model_path, render_mode=None):
        """Load a saved model and return an initialized RL2PPO instance."""
        # Temporarily suppress the FutureWarning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            # Load the saved state
            checkpoint = torch.load(model_path)
        
        # Create a new instance with the saved parameters
        instance = cls(
            env_name=checkpoint['env_name'],
            hidden_dim=checkpoint['hidden_dim'],
            render_mode=render_mode
        )
        
        # Load the saved state dictionaries
        instance.network.load_state_dict(checkpoint['network_state_dict'])
        instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training history if available
        instance.total_episodes = checkpoint.get('total_episodes', 0)
        instance.training_rewards = checkpoint.get('training_rewards', [])
        instance.eval_rewards = checkpoint.get('eval_rewards', [])
        instance.best_eval_reward = checkpoint.get('best_eval_reward', float('-inf'))
        
        # Restore directories if available
        if 'run_dir' in checkpoint:
            instance.run_dir = checkpoint['run_dir']
            instance.checkpoint_dir = checkpoint['checkpoint_dir']
            instance.log_dir = checkpoint['log_dir']
        
        return instance

    def evaluate(self, num_episodes=1, render=False, max_steps=1000):
        """Evaluate the current policy."""
        if self.rank != 0:  # Only rank 0 evaluates
            return 0.0
        
        # Use evaluation environment if rendering
        if render:
            if isinstance(self.env, CustomLunarLander):
                env = CustomLunarLander(render_mode="human")
            else:
                env = gym.make(self.env_name, render_mode="human")
        else:
            env = self.env
        
        total_reward = 0
        episode_lengths = []
        
        print(f"\nStarting evaluation over {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Set new random target for each evaluation episode
            if isinstance(env, CustomLunarLander):
                new_target = np.random.uniform(0.1, 0.9)
                env.set_target_position(new_target)
                print(f"\nEvaluation episode {episode + 1}, target: {new_target:.3f}")
            
            state = env.reset()[0]
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            hidden = torch.zeros(1, 1, self.hidden_dim)  # Initialize hidden state
            
            for step in range(max_steps):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    action_logits, _, hidden = self.network(state_tensor, hidden)
                    action_probs = torch.softmax(action_logits, dim=-1)
                    action = torch.argmax(action_probs).item()
                
                state, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if done or truncated:
                    break
                
                if step == max_steps - 1:
                    print(f"Episode {episode + 1} reached max steps limit ({max_steps})")
            
            total_reward += episode_reward
            episode_lengths.append(episode_length)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        mean_reward = total_reward / num_episodes
        mean_length = np.mean(episode_lengths)
        print(f"\nEvaluation Results:")
        print(f"Mean Reward: {mean_reward:.2f}")
        print(f"Mean Episode Length: {mean_length:.1f}")
        
        if render:
            env.close()
        
        return mean_reward

    def get_action(self, state, hidden):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            action_logits, _, new_hidden = self.network(state, hidden)
        
        dist = Categorical(logits=action_logits.squeeze())
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), new_hidden
    
    def compute_returns(self, rewards, dones, values):
        returns = []
        running_return = 0
        
        for r, d, v in zip(reversed(rewards), reversed(dones), reversed(values)):
            running_return = r + self.gamma * running_return * (1 - d)
            returns.insert(0, running_return)
            
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def collect_episode(self, max_steps=1000):
        state = self.env.reset()[0]
        hidden = torch.zeros(1, 1, self.hidden_dim)
        
        states, actions, rewards, log_probs, dones, values = [], [], [], [], [], []
        episode_reward = 0
        
        for _ in range(max_steps):
            if self.render_mode == "human":
                self.env.render()
                
            action, log_prob, new_hidden = self.get_action(state, hidden)
            next_state, reward, done, truncated, _ = self.env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(done or truncated)
            
            with torch.no_grad():
                _, value, _ = self.network(
                    torch.FloatTensor(state).unsqueeze(0).unsqueeze(0),
                    hidden
                )
                values.append(value.item())
            
            episode_reward += reward
            state = next_state
            hidden = new_hidden
            
            if done or truncated:
                break
                
        return states, actions, rewards, log_probs, dones, values, episode_reward

    def update_policy(self, all_states, all_actions, all_log_probs, all_returns):
        episodes_states = [torch.FloatTensor(np.array(states)) for states in all_states]
        episodes_actions = [torch.LongTensor(actions) for actions in all_actions]
        episodes_old_log_probs = [torch.FloatTensor(log_probs) for log_probs in all_log_probs]
        episodes_returns = [torch.FloatTensor(returns) for returns in all_returns]
        
        num_episodes = len(episodes_states)
        total_actor_loss = 0
        total_critic_loss = 0
        num_updates = 0
        
        for _ in range(self.epochs):
            episode_indices = np.random.permutation(num_episodes)
            
            for start_idx in range(0, num_episodes, self.batch_size):
                batch_indices = episode_indices[start_idx:start_idx + self.batch_size]
                actual_batch_size = len(batch_indices)
                
                max_len = max(episodes_states[idx].size(0) for idx in batch_indices)
                
                batch_states = torch.zeros(max_len, actual_batch_size, self.input_dim)
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
                
                action_logits, values, _ = self.network(batch_states)
                
                dist = Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(batch_actions)
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_returns
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_returns
                
                actor_loss = -torch.min(surr1, surr2) * batch_masks
                critic_loss = (values.squeeze(-1) - batch_returns).pow(2) * batch_masks
                
                actor_loss = actor_loss.sum() / batch_masks.sum()
                critic_loss = critic_loss.sum() / batch_masks.sum()
                
                loss = actor_loss + 0.5 * critic_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Synchronize gradients across workers
                mpi_avg_grads(self.network)
                
                self.optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                num_updates += 1
        
        # Average losses across workers
        avg_actor_loss = mpi_avg(total_actor_loss / num_updates if num_updates > 0 else 0)
        avg_critic_loss = mpi_avg(total_critic_loss / num_updates if num_updates > 0 else 0)
        return avg_actor_loss, avg_critic_loss

    def train_episode(self, num_episodes_per_update=4):
        # Set random target position for this update
        if isinstance(self.env, CustomLunarLander):
            new_target = np.random.uniform(0.1, 0.9)  # Slightly inside boundaries
            self.env.set_target_position(new_target)
            print(f"New target position: {new_target:.3f}")
        
        total_reward = 0
        total_length = 0
        
        all_states = []
        all_actions = []
        all_log_probs = []
        all_returns = []
        
        for _ in range(num_episodes_per_update):
            states, actions, rewards, log_probs, dones, values, episode_reward = self.collect_episode()
            
            returns = self.compute_returns(rewards, dones, values)
            
            all_states.append(states)
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_returns.append(returns.tolist())
            
            total_reward += episode_reward
            total_length += len(states)
        
        actor_loss, critic_loss = self.update_policy(all_states, all_actions, all_log_probs, all_returns)
        
        avg_reward = total_reward / num_episodes_per_update
        avg_length = total_length / num_episodes_per_update
        
        return avg_reward, avg_length, actor_loss, critic_loss
    
    def evaluate_saved_model(self, model_path, num_episodes=10, render=True):
        print(f"Loading model from {model_path}")
        agent = RL2PPO.load_model(model_path)
        return agent.evaluate(num_episodes=num_episodes, render=render)

class GRUNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUNetwork, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=False)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x, hidden=None):
        if hidden is None:
            batch_size = x.size(1)
            hidden = torch.zeros(1, batch_size, self.gru.hidden_size, device=x.device)
            
        output, hidden = self.gru(x, hidden)
        
        action_logits = self.policy(output)
        value = self.value(output)
        
        return action_logits, value, hidden