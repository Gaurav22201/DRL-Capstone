import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import datetime

def load_stats_data(filename):
    """Load stats data from a JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def plot_training_rewards(data, output_file='reward_plot.png'):
    """Plot the training rewards"""
    rewards = data['episode_rewards']
    
    plt.figure(figsize=(12, 6))
    
    # Plot episode rewards
    plt.plot(rewards, 'b-', alpha=0.3, label='Episode Rewards')
    
    # Calculate and plot moving average
    window_size = min(100, len(rewards))
    if len(rewards) >= window_size:
        moving_avg = [np.mean(rewards[max(0, i-window_size):i+1]) 
                     for i in range(len(rewards))]
        plt.plot(moving_avg, 'r-', label=f'{window_size}-episode Moving Average')
    
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Reward plot saved to {output_file}")

def plot_system_stats(data, output_file='system_stats_plot.png'):
    """Plot system stats (CPU, memory, GPU usage)"""
    system_stats = data['system_stats']
    
    # Extract data
    episodes = [stat['episode'] for stat in system_stats]
    timestamps = [stat['timestamp'] / 60 for stat in system_stats]  # Convert to minutes
    cpu = [stat['cpu_percent'] for stat in system_stats]
    memory = [stat['memory_percent'] for stat in system_stats]
    gpu = [stat['gpu_utilization'] for stat in system_stats]
    
    plt.figure(figsize=(12, 10))
    
    # CPU subplot
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, cpu, 'b-')
    plt.title('CPU Usage During Training')
    plt.ylabel('CPU Usage (%)')
    plt.grid(True)
    
    # Memory subplot
    plt.subplot(3, 1, 2)
    plt.plot(timestamps, memory, 'g-')
    plt.title('Memory Usage During Training')
    plt.ylabel('Memory Usage (%)')
    plt.grid(True)
    
    # GPU subplot
    plt.subplot(3, 1, 3)
    plt.plot(timestamps, gpu, 'r-')
    plt.title('GPU Usage During Training')
    plt.xlabel('Time (minutes)')
    plt.ylabel('GPU Usage (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"System stats plot saved to {output_file}")

def plot_training_time(data, output_file='training_time_plot.png'):
    """Plot episode duration over time"""
    episode_times = data['episode_times']
    
    plt.figure(figsize=(12, 6))
    
    # Plot episode times
    plt.plot(episode_times, 'g-')
    
    # Calculate and plot moving average
    window_size = min(50, len(episode_times))
    if len(episode_times) >= window_size:
        moving_avg = [np.mean(episode_times[max(0, i-window_size):i+1]) 
                     for i in range(len(episode_times))]
        plt.plot(moving_avg, 'r-', label=f'{window_size}-episode Moving Average')
    
    plt.title('Episode Duration')
    plt.xlabel('Episode')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Training time plot saved to {output_file}")

def plot_combined_stats(data, output_file='combined_stats.png'):
    """Create a combined plot of rewards and system stats"""
    rewards = data['episode_rewards']
    system_stats = data['system_stats']
    
    # Extract system stats data
    episodes = [stat['episode'] for stat in system_stats]
    cpu = [stat['cpu_percent'] for stat in system_stats]
    memory = [stat['memory_percent'] for stat in system_stats]
    gpu = [stat['gpu_utilization'] for stat in system_stats]
    
    # Make sure we have data points for each episode by interpolating if necessary
    if len(episodes) < len(rewards):
        # We need to interpolate system stats to match episode count
        all_episodes = np.arange(len(rewards))
        cpu_interp = np.interp(all_episodes, episodes, cpu)
        memory_interp = np.interp(all_episodes, episodes, memory)
        gpu_interp = np.interp(all_episodes, episodes, gpu)
    else:
        all_episodes = episodes
        cpu_interp = cpu
        memory_interp = memory
        gpu_interp = gpu
    
    plt.figure(figsize=(12, 12))
    
    # Rewards subplot
    plt.subplot(4, 1, 1)
    plt.plot(all_episodes, rewards, 'b-', alpha=0.3)
    
    # Calculate and plot moving average
    window_size = min(100, len(rewards))
    if len(rewards) >= window_size:
        moving_avg = [np.mean(rewards[max(0, i-window_size):i+1]) 
                     for i in range(len(rewards))]
        plt.plot(all_episodes, moving_avg, 'r-', 
                label=f'{window_size}-episode Moving Average')
    
    plt.title('Training Rewards')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # CPU subplot
    plt.subplot(4, 1, 2)
    plt.plot(all_episodes, cpu_interp, 'b-')
    plt.title('CPU Usage')
    plt.ylabel('CPU Usage (%)')
    plt.grid(True)
    
    # Memory subplot
    plt.subplot(4, 1, 3)
    plt.plot(all_episodes, memory_interp, 'g-')
    plt.title('Memory Usage')
    plt.ylabel('Memory Usage (%)')
    plt.grid(True)
    
    # GPU subplot
    plt.subplot(4, 1, 4)
    plt.plot(all_episodes, gpu_interp, 'r-')
    plt.title('GPU Usage')
    plt.xlabel('Episode')
    plt.ylabel('GPU Usage (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Combined stats plot saved to {output_file}")

def print_summary(data):
    """Print a summary of the training data"""
    if 'summary' in data:
        summary = data['summary']
        print("\n===== TRAINING SUMMARY =====")
        print(f"Start time: {summary['start_time']}")
        print(f"End time: {summary['end_time']}")
        
        # Format total training time
        total_time = summary['total_training_time_seconds']
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        
        print(f"Total episodes: {summary['total_episodes']}")
        print(f"Average reward (last 100 episodes): {summary['average_reward_last_100']:.2f}")
        print(f"Average episode time: {summary['average_episode_time']:.4f}s")
        print(f"Average CPU usage: {summary['average_cpu_usage']:.2f}%")
        print(f"Average memory usage: {summary['average_memory_usage']:.2f}%")
        print(f"Average GPU usage: {summary['average_gpu_usage']:.2f}%")
        print("============================")
    else:
        # Calculate summary from data
        rewards = data['episode_rewards']
        times = data['episode_times']
        system_stats = data['system_stats']
        
        cpu = [stat['cpu_percent'] for stat in system_stats]
        memory = [stat['memory_percent'] for stat in system_stats]
        gpu = [stat['gpu_utilization'] for stat in system_stats]
        
        print("\n===== TRAINING SUMMARY =====")
        print(f"Total episodes: {len(rewards)}")
        print(f"Average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")
        print(f"Average episode time: {np.mean(times):.4f}s")
        print(f"Average CPU usage: {np.mean(cpu):.2f}%")
        print(f"Average memory usage: {np.mean(memory):.2f}%")
        print(f"Average GPU usage: {np.mean(gpu):.2f}%")
        print("============================")

def main():
    parser = argparse.ArgumentParser(description='Visualize training statistics')
    parser.add_argument('stats_file', type=str, help='Path to the stats JSON file')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Directory to save output plots (default: plots)')
    parser.add_argument('--prefix', type=str, default='',
                       help='Prefix for output files')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load stats data
    data = load_stats_data(args.stats_file)
    
    # Create file prefix
    prefix = args.prefix
    if prefix and not prefix.endswith('_'):
        prefix += '_'
    
    # Generate plots
    plot_training_rewards(data, os.path.join(args.output_dir, f'{prefix}rewards.png'))
    plot_system_stats(data, os.path.join(args.output_dir, f'{prefix}system_stats.png'))
    plot_training_time(data, os.path.join(args.output_dir, f'{prefix}episode_times.png'))
    plot_combined_stats(data, os.path.join(args.output_dir, f'{prefix}combined_stats.png'))
    
    # Print summary
    print_summary(data)

if __name__ == "__main__":
    main() 