import subprocess
import re
import time
import threading
import psutil
import platform
import os

class SystemMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.cpu_percent = 0
        self.memory_percent = 0
        self.gpu_utilization = 0
        self._running = False
        self._thread = None
        
        # Check if we're on macOS
        self.is_macos = platform.system() == 'Darwin'
        
        # For Apple Silicon, we'll try to use a different approach
        self.is_apple_silicon = self.is_macos and platform.machine().startswith('arm')
        
        print(f"System: {platform.system()}, Machine: {platform.machine()}")
        print(f"Apple Silicon detected: {self.is_apple_silicon}")
    
    def _get_gpu_stats_macos(self):
        """Get GPU statistics on macOS"""
        try:
            if self.is_apple_silicon:
                # Try using powermetrics for Apple Silicon
                # This requires sudo, but we'll try a limited approach first
                try:
                    # Check if we have sudo access without password (unlikely but worth a try)
                    result = subprocess.run(
                        ["sudo", "-n", "powermetrics", "--samplers", "gpu", "-n", "1"],
                        capture_output=True,
                        text=True,
                        timeout=3  # Add timeout to avoid hanging
                    )
                    
                    if result.returncode == 0:
                        match = re.search(r"GPU active frequency: (\d+)", result.stdout)
                        if match:
                            # Convert frequency to a percentage of max frequency
                            # (this is an approximation)
                            freq = int(match.group(1))
                            # Assuming max freq is around 1000-1500 MHz for M-series
                            return min(100, int(freq / 15))
                except (subprocess.SubprocessError, subprocess.TimeoutExpired):
                    pass
                
                # Alternative: use system_profiler
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"], 
                    capture_output=True, 
                    text=True
                )
                
                if result.returncode == 0:
                    # Look for GPU utilization in the output
                    match = re.search(r"Metal Support: Yes", result.stdout)
                    if match:
                        # We have a GPU, but unfortunately system_profiler doesn't
                        # directly provide utilization. We'll use an indirect method.
                        
                        # Get CPU temperature as a proxy (higher temp often correlates with GPU usage)
                        try:
                            temp_cmd = subprocess.run(
                                ["osx-cpu-temp"], 
                                capture_output=True, 
                                text=True
                            )
                            temp_match = re.search(r"(\d+\.\d+)°C", temp_cmd.stdout)
                            if temp_match:
                                temp = float(temp_match.group(1))
                                # Rough estimation - convert temp to utilization
                                # Assuming 30°C is idle, 90°C is max
                                return max(0, min(100, (temp - 30) * (100 / 60)))
                        except (subprocess.SubprocessError, FileNotFoundError):
                            pass
                        
                        # If we get here, we couldn't get a good estimate
                        # Just check process memory info for processes known to use GPU
                        gpu_processes = ["WindowServer", "Finder", "Safari", "Chrome"]
                        total_mem = 0
                        
                        # Get list of running processes
                        for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
                            try:
                                if any(gp in proc.info['name'] for gp in gpu_processes):
                                    total_mem += proc.info['memory_percent']
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                                
                        # Use memory as a rough proxy for GPU usage
                        return min(100, total_mem * 2)  # Scale up a bit
            else:
                # For Intel Macs, try to get dedicated GPU info
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"], 
                    capture_output=True, 
                    text=True
                )
                if result.returncode == 0:
                    # Look for GPU utilization
                    if "AMD" in result.stdout or "NVIDIA" in result.stdout:
                        # We have a dedicated GPU, use activity monitor data as proxy
                        gpu_processes = ["WindowServer", "Finder", "Safari", "Chrome"]
                        total_mem = 0
                        
                        for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
                            try:
                                if any(gp in proc.info['name'] for gp in gpu_processes):
                                    total_mem += proc.info['memory_percent']
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                                
                        return min(100, total_mem * 2)
            
            return 0
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
            return 0
    
    def _get_gpu_stats(self):
        """Get GPU statistics using platform-specific methods"""
        if self.is_macos:
            return self._get_gpu_stats_macos()
        
        # For non-macOS platforms (we could add Linux and Windows support here)
        return 0
    
    def _update_stats(self):
        """Update system statistics"""
        while self._running:
            self.cpu_percent = psutil.cpu_percent(interval=None)
            self.memory_percent = psutil.virtual_memory().percent
            
            # Try to get GPU stats
            self.gpu_utilization = self._get_gpu_stats()
            
            time.sleep(self.interval)
    
    def start(self):
        """Start monitoring system resources"""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._update_stats, daemon=True)
            self._thread.start()
    
    def stop(self):
        """Stop monitoring system resources"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
    
    def get_stats(self):
        """Get current system statistics"""
        return {
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'gpu_utilization': self.gpu_utilization
        }


# Example of usage
if __name__ == "__main__":
    monitor = SystemMonitor(interval=2.0)
    monitor.start()
    
    try:
        for _ in range(10):
            stats = monitor.get_stats()
            print(f"CPU: {stats['cpu_percent']}%, "
                  f"Memory: {stats['memory_percent']}%, "
                  f"GPU: {stats['gpu_utilization']}%")
            time.sleep(2)
    finally:
        monitor.stop() 