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
            grad_numpy = param.grad.data.numpy()
            buf = np.zeros_like(grad_numpy)
            MPI.COMM_WORLD.Allreduce(grad_numpy, buf, op=MPI.SUM)
            param.grad.data = torch.from_numpy(buf / size)

def mpi_avg(x):
    """Average a float/numpy array across MPI processes."""
    if not isinstance(x, np.ndarray):
        x = np.array([x])
    buf = np.zeros_like(x)
    MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
    result = buf / MPI.COMM_WORLD.Get_size()
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
        
        # New input: concatenation of next state, one-hot action, reward, termination flag.
        self.input_dim = self.env.observation_space.shape[0] + self.env.action_space.n + 2  
        # self.input_dim = self.env.observation_space.shape[0]  # (old)
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
        if self.rank != 0:
            return None
        update_steps = self.total_episodes // 4  # assuming num_episodes_per_update=4
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
            file_list = None
            filename = f"{self.env_name}_step{update_steps}{suffix}.pth"
        
        filepath = os.path.join(save_dir, filename)
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
        
        if file_list is not None:
            file_list.append(filepath)
            if len(file_list) > keep_last_n:
                file_list.sort(key=lambda x: int(re.search(r'step(\d+)', x).group(1)))
                while len(file_list) > keep_last_n:
                    old_file = file_list.pop(0)
                    if os.path.exists(old_file):
                        os.remove(old_file)
                        print(f"Removed old file: {old_file}")
        
        print(f"Model saved to {filepath}")
        return filepath

    @classmethod
    def load_model(cls, model_path, render_mode=None):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            checkpoint = torch.load(model_path)
        
        instance = cls(
            env_name=checkpoint['env_name'],
            hidden_dim=checkpoint['hidden_dim'],
            render_mode=render_mode
        )
        
        instance.network.load_state_dict(checkpoint['network_state_dict'])
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

    def evaluate(self, num_episodes=1, render=False, max_steps=1000):
        if self.rank != 0:
            return 0.0
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
            if isinstance(env, CustomLunarLander):
                new_target = np.random.uniform(0.1, 0.9)
                env.set_target_position(new_target)
                print(f"\nEvaluation episode {episode + 1}, target: {new_target:.3f}")
            
            # Get the initial observation and initialize the policy input to zeros.
            obs = env.reset()[0]
            hidden = torch.zeros(1, 1, self.hidden_dim)
            # Initialize the policy input as the observation concatenated with zeros for (action, reward, done)
            policy_input = np.concatenate([obs, np.zeros(self.env.action_space.n + 2)])
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps):
                # Convert the current policy input to tensor and get action.
                inp_tensor = torch.FloatTensor(policy_input).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    action_logits, _, hidden = self.network(inp_tensor, hidden)
                    action_probs = torch.softmax(action_logits, dim=-1)
                    action = torch.argmax(action_probs).item()
                
                next_obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                # Create the new policy input: concat(next_obs, one-hot(action), reward, done flag)
                one_hot = np.zeros(self.env.action_space.n)
                one_hot[action] = 1
                done_flag = float(done or truncated)
                policy_input = np.concatenate([next_obs, one_hot, np.array([reward, done_flag])])
                
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

    def get_action(self, policy_input, hidden):
        # Convert the policy input vector to tensor (unsqueezed for seq and batch dims)
        inp = torch.FloatTensor(policy_input).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            action_logits, _, new_hidden = self.network(inp, hidden)
        
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
        # Get the initial observation (although not directly used in the input)
        obs = self.env.reset()[0]
        hidden = torch.zeros(1, 1, self.hidden_dim)
        # Initialize the policy input as the observation concatenated with zeros for (action, reward, done)
        policy_input = np.concatenate([obs, np.zeros(self.env.action_space.n + 2)])
        
        # Lists to store trajectories (inputs, actions, etc.)
        inputs = []      # will store the policy inputs fed into the network
        actions = []
        rewards = []
        log_probs = []
        dones = []
        values = []
        episode_reward = 0
        
        for _ in range(max_steps):
            if self.render_mode == "human":
                self.env.render()
                
            action, log_prob, new_hidden = self.get_action(policy_input, hidden)
            
            next_obs, reward, done, truncated, _ = self.env.step(action)
            
            # Record the current policy input along with action, reward, etc.
            inputs.append(policy_input)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(done or truncated)
            
            # Get the value estimate for the current step.
            with torch.no_grad():
                inp_tensor = torch.FloatTensor(policy_input).unsqueeze(0).unsqueeze(0)
                _, value, _ = self.network(inp_tensor, hidden)
                values.append(value.item())
            
            episode_reward += reward
            
            # Construct the new policy input: concatenate next_obs, one-hot(action), reward, done flag.
            one_hot = np.zeros(self.env.action_space.n)
            one_hot[action] = 1
            done_flag = float(done or truncated)
            new_policy_input = np.concatenate([next_obs, one_hot, np.array([reward, done_flag])])
            
            policy_input = new_policy_input
            hidden = new_hidden
            
            if done or truncated:
                break
                
        return inputs, actions, rewards, log_probs, dones, values, episode_reward

    def update_policy(self, all_inputs, all_actions, all_log_probs, all_returns):
        # Convert lists of episode trajectories to tensors.
        episodes_inputs = [torch.FloatTensor(np.array(episode)) for episode in all_inputs]
        episodes_actions = [torch.LongTensor(episode) for episode in all_actions]
        episodes_old_log_probs = [torch.FloatTensor(episode) for episode in all_log_probs]
        episodes_returns = [torch.FloatTensor(episode) for episode in all_returns]
        
        num_episodes = len(episodes_inputs)
        total_actor_loss = 0
        total_critic_loss = 0
        num_updates = 0
        
        for _ in range(self.epochs):
            episode_indices = np.random.permutation(num_episodes)
            for start_idx in range(0, num_episodes, self.batch_size):
                batch_indices = episode_indices[start_idx:start_idx + self.batch_size]
                actual_batch_size = len(batch_indices)
                max_len = max(episodes_inputs[idx].size(0) for idx in batch_indices)
                
                batch_inputs = torch.zeros(max_len, actual_batch_size, self.input_dim)
                batch_actions = torch.zeros(max_len, actual_batch_size, dtype=torch.long)
                batch_old_log_probs = torch.zeros(max_len, actual_batch_size)
                batch_returns = torch.zeros(max_len, actual_batch_size)
                batch_masks = torch.zeros(max_len, actual_batch_size)
                
                for batch_idx, episode_idx in enumerate(batch_indices):
                    episode_len = episodes_inputs[episode_idx].size(0)
                    batch_inputs[:episode_len, batch_idx] = episodes_inputs[episode_idx]
                    batch_actions[:episode_len, batch_idx] = episodes_actions[episode_idx]
                    batch_old_log_probs[:episode_len, batch_idx] = episodes_old_log_probs[episode_idx]
                    batch_returns[:episode_len, batch_idx] = episodes_returns[episode_idx]
                    batch_masks[:episode_len, batch_idx] = 1
                
                action_logits, values, _ = self.network(batch_inputs)
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
                mpi_avg_grads(self.network)
                self.optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                num_updates += 1
        
        avg_actor_loss = mpi_avg(total_actor_loss / num_updates if num_updates > 0 else 0)
        avg_critic_loss = mpi_avg(total_critic_loss / num_updates if num_updates > 0 else 0)
        return avg_actor_loss, avg_critic_loss

    def train_episode(self, num_episodes_per_update=4):
        if isinstance(self.env, CustomLunarLander):
            new_target = np.random.uniform(0.1, 0.9)
            self.env.set_target_position(new_target)
            print(f"New target position: {new_target:.3f}")
        
        total_reward = 0
        total_length = 0
        
        all_inputs = []
        all_actions = []
        all_log_probs = []
        all_returns = []
        
        for _ in range(num_episodes_per_update):
            inputs, actions, rewards, log_probs, dones, values, episode_reward = self.collect_episode()
            returns = self.compute_returns(rewards, dones, values)
            
            all_inputs.append(inputs)
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_returns.append(returns.tolist())
            
            total_reward += episode_reward
            total_length += len(inputs)
        
        actor_loss, critic_loss = self.update_policy(all_inputs, all_actions, all_log_probs, all_returns)
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
