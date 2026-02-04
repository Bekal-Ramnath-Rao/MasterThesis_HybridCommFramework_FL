"""
Q-Learning Based Protocol Selector for Federated Learning

This module implements a Q-learning algorithm to dynamically select
the best communication protocol based on network conditions and 
system resources.

Actions: MQTT, AMQP, gRPC, QUIC, DDS
Rewards: Communication time, Success rate, Convergence, Accuracy, Resources
Environment: Network conditions, Resources, Model size, Mobility
"""

import numpy as np
import json
import os
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pickle


class QLearningProtocolSelector:
    """
    Q-Learning agent for selecting optimal communication protocol
    """
    
    # Protocol actions
    PROTOCOLS = ['mqtt', 'amqp', 'grpc', 'quic', 'dds']
    
    # Environment state dimensions
    NETWORK_CONDITIONS = ['excellent', 'good', 'moderate', 'poor', 'very_poor']
    RESOURCE_LEVELS = ['high', 'medium', 'low']
    MODEL_SIZES = ['small', 'medium', 'large']
    MOBILITY_LEVELS = ['static', 'low', 'medium', 'high']
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        save_path: str = "q_table.pkl"
    ):
        """
        Initialize Q-Learning Protocol Selector
        
        Args:
            learning_rate: Learning rate (alpha) for Q-learning
            discount_factor: Discount factor (gamma) for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            save_path: Path to save/load Q-table
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.save_path = save_path
        
        # Initialize Q-table
        # State space: (network, resource, model_size, mobility)
        state_space_size = (
            len(self.NETWORK_CONDITIONS),
            len(self.RESOURCE_LEVELS),
            len(self.MODEL_SIZES),
            len(self.MOBILITY_LEVELS)
        )
        self.q_table = np.zeros(state_space_size + (len(self.PROTOCOLS),))
        
        # Statistics tracking
        self.episode_count = 0
        self.total_rewards = []
        self.protocol_usage = {p: 0 for p in self.PROTOCOLS}
        self.protocol_success = {p: 0 for p in self.PROTOCOLS}
        self.protocol_failures = {p: 0 for p in self.PROTOCOLS}

        # Load existing Q-table if available (will reset if dimensions don't match)
        self.load_q_table()
        
        # History for learning
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
    def get_state_index(self, state: Dict) -> Tuple[int, int, int, int]:
        """
        Convert state dictionary to indices for Q-table
        
        Args:
            state: Dictionary with network, resource, model_size, mobility
            
        Returns:
            Tuple of indices for Q-table access
        """
        network_idx = self.NETWORK_CONDITIONS.index(state.get('network', 'moderate'))
        resource_idx = self.RESOURCE_LEVELS.index(state.get('resource', 'medium'))
        model_idx = self.MODEL_SIZES.index(state.get('model_size', 'medium'))
        mobility_idx = self.MOBILITY_LEVELS.index(state.get('mobility', 'static'))
        
        return (network_idx, resource_idx, model_idx, mobility_idx)
    
    def select_protocol(self, state: Dict, training: bool = True) -> str:
        """
        Select a protocol using epsilon-greedy strategy
        
        Args:
            state: Current environment state
            training: If True, use epsilon-greedy; if False, use greedy
            
        Returns:
            Selected protocol name
        """
        state_idx = self.get_state_index(state)
        
        # Epsilon-greedy action selection
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            action_idx = np.random.randint(len(self.PROTOCOLS))
        else:
            # Exploit: best known action
            action_idx = np.argmax(self.q_table[state_idx])
        
        protocol = self.PROTOCOLS[action_idx]
        self.protocol_usage[protocol] += 1
        
        # Store for learning
        self.state_history.append(state_idx)
        self.action_history.append(action_idx)
        
        return protocol
    
    def calculate_reward(
        self,
        communication_time: float,
        success: bool,
        convergence_time: float,
        accuracy: float,
        resource_consumption: Dict[str, float]
    ) -> float:
        """
        Calculate reward based on multiple metrics
        
        Args:
            communication_time: Time for round-trip communication (seconds)
            success: Whether communication was successful
            convergence_time: Time for model convergence (seconds)
            accuracy: Model accuracy (0-1)
            resource_consumption: Dict with cpu, memory, bandwidth usage
            
        Returns:
            Calculated reward value
        """
        # Base reward for successful communication
        if not success:
            return -10.0  # Large penalty for failure
        
        reward = 10.0  # Base reward for success
        
        # 1. Communication time reward (faster is better)
        # Normalize: 0-5 seconds -> reward 5 to 0
        time_reward = max(0, 5.0 - communication_time)
        reward += time_reward
        
        # 2. Convergence time reward (faster is better)
        # Normalize: 0-100 seconds -> reward 5 to 0
        conv_reward = max(0, 5.0 - (convergence_time / 20.0))
        reward += conv_reward
        
        # 3. Accuracy reward (higher is better)
        # Scale accuracy (0-1) to reward (0-10)
        accuracy_reward = accuracy * 10.0
        reward += accuracy_reward
        
        # 4. Resource consumption penalty (lower is better)
        cpu_usage = resource_consumption.get('cpu', 0.5)
        memory_usage = resource_consumption.get('memory', 0.5)
        bandwidth_usage = resource_consumption.get('bandwidth', 0.5)
        
        # Average resource usage (0-1) converted to penalty (0 to -5)
        avg_resource = (cpu_usage + memory_usage + bandwidth_usage) / 3.0
        resource_penalty = -5.0 * avg_resource
        reward += resource_penalty
        
        return reward
    
    def update_q_value(
        self,
        reward: float,
        next_state: Optional[Dict] = None,
        done: bool = False
    ):
        """
        Update Q-value using Q-learning update rule
        
        Args:
            reward: Reward received
            next_state: Next state (None if episode ended)
            done: Whether episode is complete
        """
        if not self.state_history or not self.action_history:
            return
        
        state_idx = self.state_history[-1]
        action_idx = self.action_history[-1]
        
        # Current Q-value
        current_q = self.q_table[state_idx][action_idx]
        
        # Calculate new Q-value
        if done or next_state is None:
            # Terminal state
            new_q = current_q + self.learning_rate * (reward - current_q)
        else:
            # Non-terminal state: use Bellman equation
            next_state_idx = self.get_state_index(next_state)
            max_next_q = np.max(self.q_table[next_state_idx])
            new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * max_next_q - current_q
            )
        
        # Update Q-table
        self.q_table[state_idx][action_idx] = new_q
        
        # Store reward
        self.reward_history.append(reward)
        
        # Track protocol success/failure
        protocol = self.PROTOCOLS[action_idx]
        if reward > 0:
            self.protocol_success[protocol] += 1
        else:
            self.protocol_failures[protocol] += 1
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def end_episode(self):
        """Mark end of episode and update statistics"""
        self.episode_count += 1
        
        if self.reward_history:
            episode_reward = sum(self.reward_history[-10:])  # Last 10 rewards
            self.total_rewards.append(episode_reward)
        
        # Decay epsilon
        self.decay_epsilon()
        
        # Save Q-table periodically
        if self.episode_count % 10 == 0:
            self.save_q_table()
    
    def save_q_table(self):
        """Save Q-table and statistics to disk"""
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'total_rewards': self.total_rewards,
            'protocol_usage': self.protocol_usage,
            'protocol_success': self.protocol_success,
            'protocol_failures': self.protocol_failures
        }
        
        try:
            with open(self.save_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"[Q-Learning] Saved Q-table to {self.save_path}")
        except Exception as e:
            print(f"[Q-Learning] Error saving Q-table: {e}")
    
    def load_q_table(self):
        """Load Q-table and statistics from disk"""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'rb') as f:
                    data = pickle.load(f)
                
                loaded_q_table = data.get('q_table')
                
                # Check if dimensions match current protocol list
                expected_shape = (
                    len(self.NETWORK_CONDITIONS),
                    len(self.RESOURCE_LEVELS),
                    len(self.MODEL_SIZES),
                    len(self.MOBILITY_LEVELS),
                    len(self.PROTOCOLS)
                )
                
                if loaded_q_table is not None and loaded_q_table.shape == expected_shape:
                    self.q_table = loaded_q_table
                    self.epsilon = data.get('epsilon', self.epsilon)
                    self.episode_count = data.get('episode_count', 0)
                    self.total_rewards = data.get('total_rewards', [])
                    self.protocol_usage = data.get('protocol_usage', self.protocol_usage)
                    self.protocol_success = data.get('protocol_success', self.protocol_success)
                    self.protocol_failures = data.get('protocol_failures', self.protocol_failures)
                    
                    print(f"[Q-Learning] Loaded Q-table from {self.save_path}")
                    print(f"[Q-Learning] Episodes: {self.episode_count}, Epsilon: {self.epsilon:.4f}")
                else:
                    if loaded_q_table is not None:
                        print(f"[Q-Learning] Q-table shape mismatch: expected {expected_shape}, got {loaded_q_table.shape}")
                    print(f"[Q-Learning] Starting with fresh Q-table for {len(self.PROTOCOLS)} protocols: {self.PROTOCOLS}")
            except Exception as e:
                print(f"[Q-Learning] Error loading Q-table: {e}")
                print(f"[Q-Learning] Starting with fresh Q-table")
    
    def get_statistics(self) -> Dict:
        """Get learning statistics"""
        stats = {
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'avg_reward': np.mean(self.total_rewards[-100:]) if self.total_rewards else 0,
            'protocol_usage': self.protocol_usage,
            'protocol_success': self.protocol_success,
            'protocol_failures': self.protocol_failures,
            'success_rates': {}
        }
        
        # Calculate success rates
        for protocol in self.PROTOCOLS:
            total = self.protocol_usage[protocol]
            if total > 0:
                success_rate = self.protocol_success[protocol] / total
                stats['success_rates'][protocol] = success_rate
            else:
                stats['success_rates'][protocol] = 0.0
        
        return stats
    
    def print_statistics(self):
        """Print learning statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("Q-LEARNING PROTOCOL SELECTOR - STATISTICS")
        print("="*70)
        print(f"Episodes: {stats['episode_count']}")
        print(f"Epsilon (exploration): {stats['epsilon']:.4f}")
        print(f"Average Reward (last 100): {stats['avg_reward']:.2f}")
        print("\nProtocol Usage:")
        for protocol in self.PROTOCOLS:
            usage = stats['protocol_usage'][protocol]
            success = stats['protocol_success'][protocol]
            failure = stats['protocol_failures'][protocol]
            success_rate = stats['success_rates'][protocol]
            print(f"  {protocol.upper()}: {usage} times | Success: {success} | "
                  f"Failure: {failure} | Rate: {success_rate:.2%}")
        print("="*70 + "\n")
    
    def get_best_protocol(self, state: Dict) -> str:
        """
        Get the best protocol for given state (pure exploitation)
        
        Args:
            state: Current environment state
            
        Returns:
            Best protocol name
        """
        state_idx = self.get_state_index(state)
        action_idx = np.argmax(self.q_table[state_idx])
        return self.PROTOCOLS[action_idx]
    
    def reset_episode(self):
        """Reset episode history"""
        self.state_history = []
        self.action_history = []


class EnvironmentStateManager:
    """
    Manages environment state for RL agent
    Tracks network conditions, resources, model size, and mobility
    """
    
    def __init__(self):
        self.current_state = {
            'network': 'moderate',
            'resource': 'medium',
            'model_size': 'medium',
            'mobility': 'static'
        }
        
        # Resource monitoring
        self.cpu_usage_history = []
        self.memory_usage_history = []
        self.bandwidth_usage_history = []
        
    def update_network_condition(self, condition: str):
        """Update network condition"""
        if condition in QLearningProtocolSelector.NETWORK_CONDITIONS:
            self.current_state['network'] = condition
    
    def update_resource_level(self, level: str):
        """Update resource availability level"""
        if level in QLearningProtocolSelector.RESOURCE_LEVELS:
            self.current_state['resource'] = level
    
    def update_model_size(self, size: str):
        """Update model size"""
        if size in QLearningProtocolSelector.MODEL_SIZES:
            self.current_state['model_size'] = size
    
    def update_mobility(self, mobility: str):
        """Update mobility level"""
        if mobility in QLearningProtocolSelector.MOBILITY_LEVELS:
            self.current_state['mobility'] = mobility
    
    def detect_network_condition(self, latency_ms: float, bandwidth_mbps: float) -> str:
        """
        Detect network condition based on latency and bandwidth
        
        Args:
            latency_ms: Network latency in milliseconds
            bandwidth_mbps: Available bandwidth in Mbps
            
        Returns:
            Network condition string
        """
        # Classification based on latency and bandwidth
        if latency_ms < 10 and bandwidth_mbps > 50:
            return 'excellent'
        elif latency_ms < 30 and bandwidth_mbps > 20:
            return 'good'
        elif latency_ms < 100 and bandwidth_mbps > 5:
            return 'moderate'
        elif latency_ms < 300 and bandwidth_mbps > 1:
            return 'poor'
        else:
            return 'very_poor'
    
    def detect_resource_level(self, cpu_percent: float, memory_percent: float) -> str:
        """
        Detect resource availability based on CPU and memory usage
        
        Args:
            cpu_percent: CPU usage percentage (0-100)
            memory_percent: Memory usage percentage (0-100)
            
        Returns:
            Resource level string
        """
        avg_usage = (cpu_percent + memory_percent) / 2.0
        
        if avg_usage < 30:
            return 'high'
        elif avg_usage < 70:
            return 'medium'
        else:
            return 'low'
    
    def get_current_state(self) -> Dict:
        """Get current environment state"""
        return self.current_state.copy()
    
    def get_resource_consumption(self) -> Dict[str, float]:
        """
        Get current resource consumption metrics
        
        Returns:
            Dictionary with normalized (0-1) resource usage
        """
        try:
            import psutil
            
            cpu = psutil.cpu_percent(interval=0.1) / 100.0
            memory = psutil.virtual_memory().percent / 100.0
            
            # Bandwidth estimation (simplified)
            net_io = psutil.net_io_counters()
            bandwidth = min(1.0, (net_io.bytes_sent + net_io.bytes_recv) / 1e9)
            
            return {
                'cpu': cpu,
                'memory': memory,
                'bandwidth': bandwidth
            }
        except:
            return {'cpu': 0.5, 'memory': 0.5, 'bandwidth': 0.5}


if __name__ == "__main__":
    # Example usage
    print("Q-Learning Protocol Selector - Test")
    
    # Initialize selector
    selector = QLearningProtocolSelector(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0
    )
    
    # Initialize environment
    env_manager = EnvironmentStateManager()
    
    # Simulate training episodes
    for episode in range(50):
        print(f"\nEpisode {episode + 1}")
        
        # Random environment state
        env_manager.update_network_condition(
            np.random.choice(QLearningProtocolSelector.NETWORK_CONDITIONS)
        )
        env_manager.update_resource_level(
            np.random.choice(QLearningProtocolSelector.RESOURCE_LEVELS)
        )
        
        state = env_manager.get_current_state()
        print(f"State: {state}")
        
        # Select protocol
        protocol = selector.select_protocol(state)
        print(f"Selected protocol: {protocol}")
        
        # Simulate reward (random for demo)
        success = np.random.random() > 0.2
        comm_time = np.random.uniform(0.1, 5.0)
        conv_time = np.random.uniform(10, 100)
        accuracy = np.random.uniform(0.7, 0.95) if success else 0.0
        resources = env_manager.get_resource_consumption()
        
        reward = selector.calculate_reward(
            comm_time, success, conv_time, accuracy, resources
        )
        print(f"Reward: {reward:.2f}")
        
        # Update Q-value
        selector.update_q_value(reward, done=True)
        selector.end_episode()
    
    # Print final statistics
    selector.print_statistics()
