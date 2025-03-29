import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time
import os
import json
import matplotlib.pyplot as plt
from system_monitor import SystemMonitor  # Import the system monitor

class DQN(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(DQN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.FloatTensor(np.float32(state)), 
                torch.LongTensor(action), 
                torch.FloatTensor(reward), 
                torch.FloatTensor(np.float32(next_state)),
                torch.FloatTensor(done))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, env, learning_rate=1e-4, gamma=0.99, buffer_size=10000, 
                 batch_size=32, epsilon_start=1.0, epsilon_final=0.01, 
                 epsilon_decay=10000):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(env.observation_space.shape[0], 
                             env.action_space.n).to(self.device)
        self.target_net = DQN(env.observation_space.shape[0], 
                             env.action_space.n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        
        self.batch_size = batch_size
        self.gamma = gamma
        
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.frame_idx = 0
        
    def get_epsilon(self):
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
               np.exp(-1. * self.frame_idx / self.epsilon_decay)
    
    def select_action(self, state):
        epsilon = self.get_epsilon()
        self.frame_idx += 1
        
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.env.action_space.n)
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)
        
        current_q_values = self.policy_net(state).gather(1, action.unsqueeze(1))
        next_q_values = self.target_net(next_state).max(1)[0].detach()
        expected_q_values = reward + (1 - done) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        checkpoint = {
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'frame_idx': self.frame_idx
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.frame_idx = checkpoint['frame_idx']

def plot_training_results(rewards, moving_avg_window=100, filename='training_results.png'):
    """Plot training rewards and moving average"""
    plt.figure(figsize=(12, 8))
    
    # Plot episode rewards
    plt.plot(rewards, 'b-', alpha=0.3, label='Episode Rewards')
    
    # Calculate and plot moving average
    if len(rewards) >= moving_avg_window:
        moving_avg = [np.mean(rewards[max(0, i-moving_avg_window):i+1]) 
                      for i in range(len(rewards))]
        plt.plot(moving_avg, 'r-', label=f'{moving_avg_window}-episode Moving Average')
    
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    plt.savefig(filename)
    plt.close()

def plot_system_stats(timestamps, cpu_usage, memory_usage, gpu_usage, filename='system_stats.png'):
    """Plot system statistics over time"""
    plt.figure(figsize=(12, 10))
    
    # Convert timestamps to minutes for better readability
    minutes = [t / 60 for t in timestamps]
    
    # CPU subplot
    plt.subplot(3, 1, 1)
    plt.plot(minutes, cpu_usage, 'b-')
    plt.title('CPU Usage During Training')
    plt.ylabel('CPU Usage (%)')
    plt.grid(True)
    
    # Memory subplot
    plt.subplot(3, 1, 2)
    plt.plot(minutes, memory_usage, 'g-')
    plt.title('Memory Usage During Training')
    plt.ylabel('Memory Usage (%)')
    plt.grid(True)
    
    # GPU subplot
    plt.subplot(3, 1, 3)
    plt.plot(minutes, gpu_usage, 'r-')
    plt.title('GPU Usage During Training')
    plt.xlabel('Time (minutes)')
    plt.ylabel('GPU Usage (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def evaluate_agent(agent, env, num_episodes=5, render=True):
    """Evaluate the trained agent with visualization"""
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Render the environment if requested
            if render:
                env.render()
                time.sleep(0.01)  # Short delay to make visualization viewable
                
            # Select action without exploration
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                action = agent.policy_net(state_tensor).max(1)[1].item()
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
        print(f"Evaluation Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}")
        total_rewards.append(episode_reward)
    
    avg_reward = np.mean(total_rewards)
    print(f"Average Evaluation Reward: {avg_reward:.2f}")
    
    if render:
        env.close()
    
    return avg_reward

def main():
    # Create environment
    env = gym.make("MountainCar-v0")
    
    # Initialize agent
    agent = DQNAgent(env, learning_rate=1e-3, gamma=0.99, epsilon_decay=5000)
    
    # Initialize system monitor
    system_monitor = SystemMonitor(interval=5.0)
    system_monitor.start()
    
    # Training parameters
    episodes = 500  # Reduced episodes since MountainCar can be solved faster
    target_update = 10
    print_every = 1
    system_stats_every = 10     # Print system stats every X episodes
    progress_every = 50         # Show progress indicator every X episodes
    checkpoint_every = 50       # Save model checkpoint every X episodes
    eval_every = 100            # Evaluate with rendering every X episodes
    
    # Create checkpoint directory
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    # Create stats directory for saving data
    stats_dir = 'stats'
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    
    # Performance tracking
    episode_durations = []
    episode_times = []
    system_stats_history = []
    episode_numbers = []  # Track episode numbers for system stats
    timestamps = []       # Track timestamps for system stats
    cpu_usage = []        # Track CPU usage
    memory_usage = []     # Track memory usage
    gpu_usage = []        # Track GPU usage
    
    print(f"Training on device: {agent.device}")
    print(f"Starting training for {episodes} episodes...")
    
    # Track total training time
    total_start_time = time.time()
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Training loop
    for episode in range(episodes):
        start_time = time.time()
        state, _ = env.reset()
        episode_reward = 0
        
        # Capture system stats at the start of each episode
        stats = system_monitor.get_stats()
        episode_numbers.append(episode)
        current_timestamp = time.time() - total_start_time
        timestamps.append(current_timestamp)
        cpu_usage.append(stats['cpu_percent'])
        memory_usage.append(stats['memory_percent'])
        gpu_usage.append(stats['gpu_utilization'])
        
        # Add current stats to history with timestamp
        system_stats_history.append({
            'episode': episode,
            'timestamp': current_timestamp,
            'cpu_percent': stats['cpu_percent'],
            'memory_percent': stats['memory_percent'],
            'gpu_utilization': stats['gpu_utilization']
        })
        
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            loss = agent.train_step()
            
            if done:
                break
            
            # Update target network
            if agent.frame_idx % target_update == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
        
        # Track episode statistics
        episode_time = time.time() - start_time
        episode_times.append(episode_time)
        episode_durations.append(episode_reward)
        
        if episode % print_every == 0:
            print(f"Episode {episode}, Total reward: {episode_reward}, " 
                  f"Epsilon: {agent.get_epsilon():.3f}, Time: {episode_time:.2f}s")
            
        # Track and print system stats periodically
        if episode % system_stats_every == 0:
            stats = system_monitor.get_stats()
            print(f"System Stats - CPU: {stats['cpu_percent']:.1f}%, "
                  f"Memory: {stats['memory_percent']:.1f}%, "
                  f"GPU: {stats['gpu_utilization']}%")
                  
        # Show progress indicator
        if episode % progress_every == 0 and episode > 0:
            progress_pct = (episode / episodes) * 100
            elapsed_time = time.time() - total_start_time
            estimated_total = elapsed_time / (episode / episodes)
            estimated_remaining = estimated_total - elapsed_time
            
            # Format remaining time
            hours_left, remainder = divmod(estimated_remaining, 3600)
            minutes_left, seconds_left = divmod(remainder, 60)
            
            print(f"Progress: {progress_pct:.1f}% complete, "
                  f"Estimated time remaining: {int(hours_left)}h {int(minutes_left)}m {seconds_left:.0f}s")
        
        # Save checkpoint only (no plots)
        if episode % checkpoint_every == 0 and episode > 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_episode_{episode}.pt')
            agent.save_checkpoint(checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Evaluate agent with visualization periodically
        if episode % eval_every == 0 and episode > 0:
            print("\nEvaluating agent with visualization...")
            # Create a separate environment for rendering
            eval_env = gym.make("MountainCar-v0", render_mode="human")
            evaluate_agent(agent, eval_env)
            eval_env.close()
            print("Evaluation complete, continuing training...\n")
    
    # Calculate total training time
    total_training_time = time.time() - total_start_time
    end_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Stop the system monitor when done
    system_monitor.stop()
    
    # Format total training time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\nTraining completed!")
    print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Average reward over last 100 episodes: {np.mean(episode_durations[-100:]):.2f}")
    print(f"Average episode time: {np.mean(episode_times):.2f}s")
    
    # Print average system stats
    if system_stats_history:
        avg_cpu = np.mean(cpu_usage)
        avg_memory = np.mean(memory_usage)
        avg_gpu = np.mean(gpu_usage)
        print(f"Average System Usage - CPU: {avg_cpu:.1f}%, "
              f"Memory: {avg_memory:.1f}%, GPU: {avg_gpu:.1f}%")
    
    # Final evaluation with visualization
    print("\nFinal agent evaluation with visualization:")
    final_eval_env = gym.make("MountainCar-v0", render_mode="human")
    final_reward = evaluate_agent(agent, final_eval_env)
    final_eval_env.close()
    
    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, 'final_model.pt')
    agent.save_checkpoint(final_checkpoint_path)
    print(f"Final model saved to {final_checkpoint_path}")
    
    # Save final system stats data
    final_stats_file = os.path.join(stats_dir, 'final_stats.json')
    
    # Add summary data
    summary_data = {
        'start_time': start_timestamp,
        'end_time': end_timestamp,
        'total_training_time_seconds': total_training_time,
        'total_episodes': episodes,
        'average_reward_last_100': float(np.mean(episode_durations[-100:])),
        'average_episode_time': float(np.mean(episode_times)),
        'average_cpu_usage': float(avg_cpu),
        'average_memory_usage': float(avg_memory),
        'average_gpu_usage': float(avg_gpu)
    }
    
    save_stats_data(
        system_stats_history,
        episode_durations,
        episode_times,
        final_stats_file,
        summary=summary_data
    )
    print(f"Final stats saved to {final_stats_file}")
    
    # Plot final training results
    plot_training_results(episode_durations)
    
    # Plot final system stats
    plot_system_stats(timestamps, cpu_usage, memory_usage, gpu_usage)
    
    print("Final training results and system stats plots saved")

def save_stats_data(system_stats, rewards, times, filename, summary=None):
    """Save system statistics and training data to a JSON file"""
    data = {
        'system_stats': system_stats,
        'episode_rewards': rewards,
        'episode_times': times,
    }
    
    # Add summary if provided
    if summary:
        data['summary'] = summary
        
    # Convert NumPy values to Python types for JSON serialization
    data_serializable = json.loads(
        json.dumps(data, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    )
    
    with open(filename, 'w') as f:
        json.dump(data_serializable, f, indent=2)

if __name__ == "__main__":
    main() 