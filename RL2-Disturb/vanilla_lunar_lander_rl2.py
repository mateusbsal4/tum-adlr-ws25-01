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
import logging
import argparse
import torch.nn.utils as nn_utils
from mpi4py import MPI

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
    def __init__(self, env_name, hidden_dim=128, lr=3e-4, gamma=0.99, epsilon=0.2, 
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
        
        # Create environment for each worker
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
        """Save the model with a timestamp and optional suffix."""
        if self.rank != 0:
            return None
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Determine the save directory and file list based on suffix
        if suffix == '_checkpoint':
            save_dir = self.checkpoint_dir
            file_list = self.checkpoint_files
        elif suffix == '_best':
            save_dir = self.run_dir
            file_list = self.best_model_files
        else:
            save_dir = self.run_dir
            file_list = None  # Don't manage other types of saves
        
        filename = f"{self.env_name}_{timestamp}{suffix}.pth"
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
                old_file = file_list.pop(0)
                if os.path.exists(old_file):
                    os.remove(old_file)
                    print(f"Removed old file: {old_file}")
        
        print(f"Model saved to {filepath}")
        return filepath

    @classmethod
    def load_model(cls, model_path, render_mode=None):
        """Load a saved model and return an initialized RL2PPO instance."""
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

    def evaluate_model(self, num_episodes=10, render=False):
        """Evaluate the model with optional rendering."""
        if render:
            env = gym.make(self.env_name, render_mode="human")
        else:
            env = self.env
        
        eval_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state = env.reset()[0]
            hidden = torch.zeros(1, 1, self.hidden_dim)
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _, new_hidden = self.get_action(state, hidden)
                state, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                hidden = new_hidden
                done = done or truncated
                episode_length += 1
            
            eval_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        mean_length = np.mean(episode_lengths)
        
        print("\nEvaluation Results:")
        print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Mean Episode Length: {mean_length:.1f}")
        
        return mean_reward, std_reward, mean_length

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
    
    def evaluate(self, num_episodes=10):
        """Evaluate the model's performance."""
        # Only evaluate on rank 0
        if self.rank == 0:
            eval_rewards = []
            
            for _ in range(num_episodes):
                state = self.env.reset()[0]
                hidden = torch.zeros(1, 1, self.hidden_dim)
                episode_reward = 0
                done = False
                
                while not done:
                    action, _, new_hidden = self.get_action(state, hidden)
                    state, reward, done, truncated, _ = self.env.step(action)
                    episode_reward += reward
                    hidden = new_hidden
                    done = done or truncated
                
                eval_rewards.append(episode_reward)
            
            return np.mean(eval_rewards)
        return 0.0

def plot_rewards(training_rewards, eval_rewards, eval_interval, num_episodes_per_update, save_dir):
    if len(training_rewards) == 0 or len(eval_rewards) == 0:
        return None  # Don't create plot if no data
        
    plt.figure(figsize=(12, 6))
    
    # Plot training rewards
    training_episodes = [i * num_episodes_per_update for i in range(len(training_rewards))]
    plt.plot(training_episodes, training_rewards, 
             label='Training Rewards', alpha=0.6)
    
    # Calculate evaluation episodes correctly
    eval_episodes = []
    current_episode = 0
    for _ in range(len(eval_rewards)):
        eval_episodes.append(current_episode)
        current_episode += eval_interval
    
    # Ensure eval_episodes and eval_rewards have the same length
    eval_episodes = eval_episodes[:len(eval_rewards)]
    eval_rewards = eval_rewards[:len(eval_episodes)]
    
    plt.plot(eval_episodes, eval_rewards, 
             label='Evaluation Rewards', color='red', marker='o')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training and Evaluation Rewards')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(save_dir, 'rewards_plot.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def setup_logger(log_dir):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    logger = logging.getLogger('RL2PPO')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

def find_latest_model(run_dir):
    """Find the latest model file in the run directory."""
    if not os.path.exists(run_dir):
        return None
        
    model_files = []
    # Look in main run directory and checkpoints subdirectory
    for root in [run_dir, os.path.join(run_dir, 'checkpoints')]:
        if os.path.exists(root):
            for file in os.listdir(root):
                if file.endswith('.pth'):
                    full_path = os.path.join(root, file)
                    model_files.append((os.path.getmtime(full_path), full_path))
    
    if not model_files:
        return None
        
    # Return the most recent file
    return sorted(model_files, key=lambda x: x[0])[-1][1]

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate RL2-PPO on LunarLander')
    parser.add_argument('--mode', choices=['train', 'evaluate'], default='train',
                        help='Whether to train a new model or evaluate a saved one')
    parser.add_argument('--model-path', type=str, help='Path to saved model for evaluation or resuming training')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes for evaluation')
    parser.add_argument('--render', action='store_true',
                        help='Whether to render the environment during training/evaluation')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from a saved model')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Custom name for the training run directory')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume training from the latest model in specified run directory')
    
    args = parser.parse_args()
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    env_name = "LunarLander-v2"
    num_episodes = 10000  # This is now the total episodes across all workers
    eval_interval = 50
    num_episodes_per_update = 10
    num_checkpoints_to_keep = 5
    
    # Calculate episodes per worker
    episodes_per_worker = num_episodes // size
    if rank == size - 1:  # Last worker gets any remaining episodes
        episodes_per_worker += num_episodes % size
    
    # Only render on rank 0
    render_mode = "human" if args.render and rank == 0 else None
    
    if args.resume_from:
        # Find the latest model in the specified run directory
        run_dir = os.path.join('models', args.resume_from)
        latest_model = find_latest_model(run_dir)
        
        if latest_model is None and rank == 0:
            print(f"No model files found in {run_dir}")
            return
            
        agent = RL2PPO.load_model(latest_model, render_mode=render_mode)
        if rank == 0:
            logger = setup_logger(agent.log_dir)
            logger.info(f"Resuming training from {latest_model}")
        num_episodes = max(num_episodes - agent.total_episodes, 0)
        training_rewards = agent.training_rewards
        eval_rewards = agent.eval_rewards
        best_eval_reward = agent.best_eval_reward
    else:
        agent = RL2PPO(env_name, render_mode=render_mode, run_name=args.run_name)
        if rank == 0:
            logger = setup_logger(agent.log_dir)
        training_rewards = []
        eval_rewards = []
        best_eval_reward = float('-inf')
    
    reward_window = deque(maxlen=100 // num_episodes_per_update)
    
    if rank == 0:
        logger.info(f"Starting training with {size} workers...")
        logger.info("Initialization parameters:")
        logger.info(f"  Environment: {env_name}")
        logger.info(f"  Hidden dimension: {agent.hidden_dim}")
        logger.info(f"  Learning rate: {agent.optimizer.param_groups[0]['lr']}")
        logger.info(f"  Gamma (discount): {agent.gamma}")
        logger.info(f"  Epsilon (clip): {agent.epsilon}")
        logger.info(f"  PPO epochs: {agent.epochs}")
        logger.info(f"  Batch size: {agent.batch_size}")
        logger.info(f"  Run directory: {agent.run_dir}")
        logger.info("\nTraining parameters:")
        logger.info(f"  Total episodes: {num_episodes}")
        logger.info(f"  Starting from episode: {agent.total_episodes}")
        logger.info(f"  Episodes remaining: {num_episodes - agent.total_episodes}")
        logger.info(f"  Evaluation interval: {eval_interval}")
        logger.info(f"  Episodes per update: {num_episodes_per_update}")
        logger.info(f"  Checkpoints to keep: {num_checkpoints_to_keep}")
        logger.info(f"  Rendering: {args.render}")
        logger.info("-" * 50)
    
    if not args.resume:
        eval_reward = agent.evaluate()
        if rank == 0:  # Only rank 0 should append and log
            eval_rewards.append(eval_reward)
            logger.info(f"Initial evaluation reward: {eval_reward:.2f}")
            logger.info("-" * 50)
    
    try:
        while agent.total_episodes < episodes_per_worker:  # Each worker runs its share
            avg_reward, avg_length, actor_loss, critic_loss = agent.train_episode(num_episodes_per_update)
            
            # Synchronize rewards and metrics across workers
            avg_reward = float(mpi_avg(avg_reward))
            avg_length = float(mpi_avg(avg_length))
            actor_loss = float(mpi_avg(actor_loss))
            critic_loss = float(mpi_avg(critic_loss))
            
            if rank == 0:  # Only rank 0 should log
                training_rewards.append(avg_reward)
                reward_window.append(avg_reward)
                
                agent.training_rewards = training_rewards
                agent.eval_rewards = eval_rewards
                agent.best_eval_reward = best_eval_reward
                
                # Calculate global episode number for logging
                global_episode = agent.total_episodes * size  # Multiply by number of workers
                
                logger.info(f"Global Episode {global_episode}:")
                logger.info(f"  Average episode length: {avg_length:.1f}")
                logger.info(f"  Average training reward: {avg_reward:.2f}")
                logger.info(f"  Actor loss: {actor_loss:.4f}")
                logger.info(f"  Critic loss: {critic_loss:.4f}")
            
            if (agent.total_episodes * size) % eval_interval == 0 and rank == 0:
                eval_reward = agent.evaluate()
                if rank == 0:  # Only rank 0 should append and log
                    eval_rewards.append(eval_reward)
                    logger.info(f"  Evaluation reward: {eval_reward:.2f}")
                    logger.info("-" * 50)
                    
                    if eval_reward > best_eval_reward:
                        best_eval_reward = eval_reward
                        model_path = agent.save_model("_best")
                        logger.info(f"  New best model saved: {model_path}")
                    
                    plot_path = plot_rewards(training_rewards, eval_rewards, eval_interval, 
                                          num_episodes_per_update, agent.run_dir)
                    logger.info(f"  Updated rewards plot saved as '{plot_path}'")
            
            if (agent.total_episodes * size) % 200 == 0 and rank == 0:
                model_path = agent.save_model("_checkpoint")
                logger.info(f"  Checkpoint saved: {model_path}")
            
            agent.total_episodes += num_episodes_per_update
            
    except KeyboardInterrupt:
        if rank == 0:  # Only rank 0 should handle interruption
            logger.info("\nTraining interrupted by user!")
            model_path = agent.save_model("_interrupted")
            logger.info(f"Interrupted model saved: {model_path}")
    
    finally:
        if rank == 0:  # Only rank 0 should handle cleanup
            model_path = agent.save_model("_final")
            logger.info(f"Final model saved: {model_path}")
            
            plot_path = plot_rewards(training_rewards, eval_rewards, eval_interval, 
                                   num_episodes_per_update, agent.run_dir)
            logger.info(f"Training completed! Results plot saved as '{plot_path}'")

def evaluate_saved_model(model_path, num_episodes=10, render=True):
    print(f"Loading model from {model_path}")
    agent = RL2PPO.load_model(model_path)
    return agent.evaluate_model(num_episodes=num_episodes, render=render)

if __name__ == "__main__":
    main() 