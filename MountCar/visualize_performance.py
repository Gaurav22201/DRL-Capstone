import matplotlib.pyplot as plt
import numpy as np
import time
import json
import os
from system_monitor import SystemMonitor

def plot_live_stats(duration_seconds=60, save_file='system_stats.json'):
    """
    Monitor and plot system stats in real-time for a specified duration
    
    Args:
        duration_seconds: How long to monitor (in seconds)
        save_file: Where to save the recorded data
    """
    # Initialize the system monitor
    monitor = SystemMonitor(interval=1.0)
    monitor.start()
    
    # Setup data containers
    times = []
    cpu_data = []
    memory_data = []
    gpu_data = []
    
    # Setup the plot
    plt.figure(figsize=(12, 8))
    plt.ion()  # Interactive mode on
    
    start_time = time.time()
    try:
        while time.time() - start_time < duration_seconds:
            # Get current stats
            stats = monitor.get_stats()
            current_time = time.time() - start_time
            
            # Store the data
            times.append(current_time)
            cpu_data.append(stats['cpu_percent'])
            memory_data.append(stats['memory_percent'])
            gpu_data.append(stats['gpu_utilization'])
            
            # Clear and redraw the plot
            plt.clf()
            
            # CPU subplot
            plt.subplot(3, 1, 1)
            plt.plot(times, cpu_data, 'b-')
            plt.title('CPU Usage')
            plt.ylabel('Usage (%)')
            plt.grid(True)
            
            # Memory subplot
            plt.subplot(3, 1, 2)
            plt.plot(times, memory_data, 'g-')
            plt.title('Memory Usage')
            plt.ylabel('Usage (%)')
            plt.grid(True)
            
            # GPU subplot
            plt.subplot(3, 1, 3)
            plt.plot(times, gpu_data, 'r-')
            plt.title('GPU Usage')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Usage (%)')
            plt.grid(True)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)
            
            # Small pause to not overwhelm the system
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("Monitoring stopped by user")
    finally:
        monitor.stop()
        
        # Save the data
        data = {
            'times': times,
            'cpu': cpu_data,
            'memory': memory_data,
            'gpu': gpu_data
        }
        
        with open(save_file, 'w') as f:
            json.dump(data, f)
        
        print(f"Data saved to {save_file}")
        
        # Final plot - non-interactive for clear viewing
        plt.ioff()
        
        plt.figure(figsize=(12, 8))
        
        # CPU subplot
        plt.subplot(3, 1, 1)
        plt.plot(times, cpu_data, 'b-')
        plt.title('CPU Usage')
        plt.ylabel('Usage (%)')
        plt.grid(True)
        
        # Memory subplot
        plt.subplot(3, 1, 2)
        plt.plot(times, memory_data, 'g-')
        plt.title('Memory Usage')
        plt.ylabel('Usage (%)')
        plt.grid(True)
        
        # GPU subplot
        plt.subplot(3, 1, 3)
        plt.plot(times, gpu_data, 'r-')
        plt.title('GPU Usage')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Usage (%)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('system_performance.png')
        plt.show()

def load_and_plot_stats(file_path='system_stats.json'):
    """
    Load and plot previously saved system statistics
    
    Args:
        file_path: Path to the JSON file containing system stats
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    times = data['times']
    cpu_data = data['cpu']
    memory_data = data['memory']
    gpu_data = data['gpu']
    
    plt.figure(figsize=(12, 8))
    
    # CPU subplot
    plt.subplot(3, 1, 1)
    plt.plot(times, cpu_data, 'b-')
    plt.title('CPU Usage')
    plt.ylabel('Usage (%)')
    plt.grid(True)
    
    # Memory subplot
    plt.subplot(3, 1, 2)
    plt.plot(times, memory_data, 'g-')
    plt.title('Memory Usage')
    plt.ylabel('Usage (%)')
    plt.grid(True)
    
    # GPU subplot
    plt.subplot(3, 1, 3)
    plt.plot(times, gpu_data, 'r-')
    plt.title('GPU Usage')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Usage (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor and visualize system performance')
    parser.add_argument('--duration', type=int, default=60, 
                        help='Duration to monitor system (seconds)')
    parser.add_argument('--load', type=str, default=None,
                        help='Load and visualize a saved stats file')
    
    args = parser.parse_args()
    
    if args.load:
        load_and_plot_stats(args.load)
    else:
        plot_live_stats(duration_seconds=args.duration) 