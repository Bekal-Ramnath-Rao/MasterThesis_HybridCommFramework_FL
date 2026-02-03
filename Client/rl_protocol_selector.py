"""
Reinforcement Learning-Based Protocol Selection for Federated Learning

This module implements RL agents that learn optimal protocol selection
through interaction and feedback (rewards).

Approaches:
1. Q-Learning: Learn Q-values for (state, action) pairs
2. Deep Q-Network (DQN): Use neural network to approximate Q-function
3. Multi-Armed Bandit: Simpler approach for stateless scenarios

The agent learns by:
- Observing state (network conditions, resources, etc.)
- Taking action (selecting a protocol)
- Receiving reward (based on convergence time, accuracy, cost)
- Updating policy to maximize long-term reward
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict
import random


class ProtocolState:
    """
    Represents the current state for RL decision making.
    Discretizes continuous features into bins for tabular Q-learning.
    """
    
    def __init__(
        self,
        latency: float,
        bandwidth: float,
        packet_loss: float,
        cpu_usage: float,
        memory_usage: float,
        battery_level: float,
        model_size: float,
        is_mobile: bool
    ):
        # Discretize continuous features into bins
        self.latency_bin = self._discretize(latency, [0, 50, 100, 200, 500])
        self.bandwidth_bin = self._discretize(bandwidth, [0, 1, 5, 10, 50, 100])
        self.packet_loss_bin = self._discretize(packet_loss, [0, 0.5, 1, 2, 5])
        self.cpu_bin = self._discretize(cpu_usage, [0, 30, 50, 70, 90])
        self.memory_bin = self._discretize(memory_usage, [0, 30, 50, 70, 90])
        self.battery_bin = self._discretize(battery_level, [0, 20, 40, 60, 80])
        self.model_size_bin = self._discretize(model_size, [0, 10, 50, 100, 200])
        self.is_mobile = int(is_mobile)
    
    def _discretize(self, value: float, bins: List[float]) -> int:
        """Discretize continuous value into bins"""
        for i, threshold in enumerate(bins[1:], 1):
            if value < threshold:
                return i - 1
        return len(bins) - 1
    
    def to_tuple(self) -> Tuple:
        """Convert state to hashable tuple for Q-table"""
        return (
            self.latency_bin,
            self.bandwidth_bin,
            self.packet_loss_bin,
            self.cpu_bin,
            self.memory_bin,
            self.battery_bin,
            self.model_size_bin,
            self.is_mobile
        )
    
    def __hash__(self):
        return hash(self.to_tuple())
    
    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()
    
    def __repr__(self):
        return f"State{self.to_tuple()}"


class QLearningProtocolSelector:
    """
    Q-Learning agent for protocol selection.
    
    State: Current system conditions (discretized)
    Actions: Protocol choices (mqtt, amqp, grpc, quic, dds)
    Reward: Based on convergence time, accuracy, and resource usage
    
    Q-Learning Update Rule:
    Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]
    
    Where:
    - α (alpha): Learning rate
    - γ (gamma): Discount factor (how much to value future rewards)
    - r: Immediate reward
    - s, a: Current state and action
    - s': Next state
    """
    
    def __init__(
        self,
        alpha: float = 0.1,           # Learning rate
        gamma: float = 0.95,           # Discount factor
        epsilon: float = 0.1,          # Exploration rate
        epsilon_decay: float = 0.995,  # Decay epsilon over time
        epsilon_min: float = 0.01
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: {state: {action: Q-value}}
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Actions (protocols)
        self.actions = ['mqtt', 'amqp', 'grpc', 'quic', 'dds']
        
        # Statistics
        self.episodes = 0
        self.total_reward = 0
        self.rewards_history = []
        self.action_counts = defaultdict(int)
    
    def get_state(self, conditions: Dict) -> ProtocolState:
        """Convert conditions dictionary to ProtocolState"""
        network = conditions.get('network', {})
        resources = conditions.get('resources', {})
        model = conditions.get('model', {})
        mobility = conditions.get('mobility', {})
        
        return ProtocolState(
            latency=network.get('latency', 50),
            bandwidth=network.get('bandwidth', 10),
            packet_loss=network.get('packet_loss', 0),
            cpu_usage=resources.get('cpu_usage', 50),
            memory_usage=resources.get('memory_usage', 50),
            battery_level=resources.get('battery_level', 100),
            model_size=model.get('model_size_mb', 10),
            is_mobile=mobility.get('is_mobile', False)
        )
    
    def select_action(self, state: ProtocolState, mode: str = 'train') -> str:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            mode: 'train' (epsilon-greedy) or 'exploit' (greedy only)
            
        Returns:
            Selected protocol
        """
        # Exploitation: choose best action
        if mode == 'exploit' or random.random() > self.epsilon:
            q_values = self.q_table[state]
            if not q_values:  # If state never seen, explore
                return random.choice(self.actions)
            # Choose action with highest Q-value
            best_action = max(q_values.items(), key=lambda x: x[1])[0]
            return best_action
        
        # Exploration: random action
        else:
            return random.choice(self.actions)
    
    def calculate_reward(self, performance: Dict) -> float:
        """
        Calculate reward based on FL performance.
        
        Reward components:
        - Convergence time (lower is better): -time_penalty
        - Final accuracy (higher is better): +accuracy_reward
        - Resource usage (lower is better): -resource_penalty
        
        Args:
            performance: Dict with convergence_time, final_accuracy, etc.
            
        Returns:
            Reward value
        """
        convergence_time = performance.get('convergence_time', 100)  # seconds
        final_accuracy = performance.get('final_accuracy', 0.5)
        cpu_avg = performance.get('avg_cpu_usage', 50)
        memory_avg = performance.get('avg_memory_usage', 50)
        
        # Normalize and combine components
        # Time: normalize to 0-1 scale (assume 0-200s range), invert, scale by 100
        time_reward = (200 - min(convergence_time, 200)) / 2  # 0-100
        
        # Accuracy: scale to 0-100
        accuracy_reward = final_accuracy * 100  # 0-100
        
        # Resource penalty: average of CPU and memory (inverted)
        resource_penalty = (cpu_avg + memory_avg) / 4  # 0-50
        
        # Combined reward (weighted sum)
        reward = (
            0.4 * time_reward +        # 40% weight on time
            0.5 * accuracy_reward +    # 50% weight on accuracy
            -0.1 * resource_penalty    # 10% penalty for resource usage
        )
        
        return reward
    
    def update_q_value(
        self,
        state: ProtocolState,
        action: str,
        reward: float,
        next_state: Optional[ProtocolState] = None
    ):
        """
        Update Q-value using Q-learning update rule.
        
        Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]
        """
        current_q = self.q_table[state][action]
        
        if next_state is not None:
            # Get max Q-value for next state
            next_q_values = self.q_table[next_state]
            max_next_q = max(next_q_values.values()) if next_q_values else 0
            
            # Q-learning update
            new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        else:
            # Terminal state (no next state)
            new_q = current_q + self.alpha * (reward - current_q)
        
        self.q_table[state][action] = new_q
    
    def train_episode(
        self,
        conditions: Dict,
        protocol: str,
        performance: Dict
    ):
        """
        Train on a single episode (one FL experiment).
        
        Args:
            conditions: System conditions when protocol was selected
            protocol: Protocol that was used
            performance: Results from FL experiment
        """
        state = self.get_state(conditions)
        reward = self.calculate_reward(performance)
        
        # Update Q-value
        self.update_q_value(state, protocol, reward, next_state=None)
        
        # Update statistics
        self.episodes += 1
        self.total_reward += reward
        self.rewards_history.append(reward)
        self.action_counts[protocol] += 1
        
        # Decay epsilon (explore less over time)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def select_best_protocol(
        self,
        conditions: Dict,
        verbose: bool = True
    ) -> Tuple[str, Dict]:
        """
        Select best protocol for given conditions (exploitation mode).
        
        Returns:
            (protocol, q_values_dict)
        """
        state = self.get_state(conditions)
        protocol = self.select_action(state, mode='exploit')
        
        q_values = dict(self.q_table[state])
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Q-LEARNING PROTOCOL SELECTION")
            print(f"{'='*70}")
            print(f"\n📊 State: {state}")
            print(f"\n🏆 Selected Protocol: {protocol.upper()}")
            print(f"\nQ-Values:")
            for action in self.actions:
                q_val = q_values.get(action, 0)
                indicator = "✓" if action == protocol else " "
                bar = '█' * int(max(0, q_val) / 10)
                print(f"  {indicator} {action:6s}: {q_val:7.2f} {bar}")
            print(f"\nExploration rate (ε): {self.epsilon:.4f}")
            print(f"{'='*70}\n")
        
        return protocol, q_values
    
    def get_statistics(self) -> Dict:
        """Get learning statistics"""
        avg_reward = np.mean(self.rewards_history[-100:]) if self.rewards_history else 0
        
        return {
            'episodes': self.episodes,
            'total_reward': self.total_reward,
            'avg_reward_last_100': avg_reward,
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'action_distribution': dict(self.action_counts)
        }
    
    def save(self, path: str):
        """Save Q-table and parameters"""
        save_dict = {
            'q_table': dict(self.q_table),
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'episodes': self.episodes,
            'total_reward': self.total_reward,
            'rewards_history': self.rewards_history,
            'action_counts': dict(self.action_counts)
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"✓ Q-Learning model saved to {path}")
    
    def load(self, path: str):
        """Load Q-table and parameters"""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Convert back to defaultdict
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state_tuple, actions in save_dict['q_table'].items():
            for action, q_value in actions.items():
                self.q_table[state_tuple][action] = q_value
        
        self.alpha = save_dict['alpha']
        self.gamma = save_dict['gamma']
        self.epsilon = save_dict['epsilon']
        self.epsilon_decay = save_dict['epsilon_decay']
        self.epsilon_min = save_dict['epsilon_min']
        self.episodes = save_dict['episodes']
        self.total_reward = save_dict['total_reward']
        self.rewards_history = save_dict['rewards_history']
        self.action_counts = defaultdict(int, save_dict['action_counts'])
        
        print(f"✓ Q-Learning model loaded from {path}")
        print(f"  Episodes trained: {self.episodes}")
        print(f"  Q-table size: {len(self.q_table)} states")


class MultiArmedBanditSelector:
    """
    Multi-Armed Bandit approach for protocol selection.
    Simpler than full RL - treats protocol selection as a stateless problem.
    
    Uses Upper Confidence Bound (UCB) algorithm:
    UCB(a) = μ(a) + c * sqrt(ln(t) / n(a))
    
    Where:
    - μ(a): Average reward for action a
    - t: Total number of trials
    - n(a): Number of times action a was selected
    - c: Exploration parameter
    """
    
    def __init__(self, c: float = 2.0):
        """
        Initialize bandit.
        
        Args:
            c: Exploration parameter (higher = more exploration)
        """
        self.c = c
        self.actions = ['mqtt', 'amqp']
        
        # Statistics per action
        self.counts = {action: 0 for action in self.actions}
        self.rewards = {action: [] for action in self.actions}
        self.total_trials = 0
    
    def calculate_ucb(self, action: str) -> float:
        """Calculate UCB score for an action"""
        if self.counts[action] == 0:
            return float('inf')  # Always explore untried actions first
        
        avg_reward = np.mean(self.rewards[action])
        exploration_bonus = self.c * np.sqrt(np.log(self.total_trials) / self.counts[action])
        
        return avg_reward + exploration_bonus
    
    def select_protocol(self, verbose: bool = True) -> str:
        """Select protocol using UCB algorithm"""
        ucb_scores = {action: self.calculate_ucb(action) for action in self.actions}
        best_action = max(ucb_scores.items(), key=lambda x: x[1])[0]
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"MULTI-ARMED BANDIT PROTOCOL SELECTION")
            print(f"{'='*70}")
            print(f"\n🎯 Selected Protocol: {best_action.upper()}")
            print(f"\nUCB Scores:")
            for action, score in sorted(ucb_scores.items(), key=lambda x: x[1], reverse=True):
                avg_reward = np.mean(self.rewards[action]) if self.rewards[action] else 0
                indicator = "✓" if action == best_action else " "
                print(f"  {indicator} {action:6s}: UCB={score:8.2f} | "
                      f"Avg Reward={avg_reward:6.2f} | Trials={self.counts[action]:3d}")
            print(f"\nTotal trials: {self.total_trials}")
            print(f"{'='*70}\n")
        
        return best_action
    
    def update(self, action: str, reward: float):
        """Update statistics after observing reward"""
        self.counts[action] += 1
        self.rewards[action].append(reward)
        self.total_trials += 1
    
    def get_statistics(self) -> Dict:
        """Get bandit statistics"""
        stats = {
            'total_trials': self.total_trials,
            'protocols': {}
        }
        
        for action in self.actions:
            stats['protocols'][action] = {
                'trials': self.counts[action],
                'avg_reward': np.mean(self.rewards[action]) if self.rewards[action] else 0,
                'total_reward': sum(self.rewards[action])
            }
        
        return stats
    
    def save(self, path: str):
        """Save bandit state"""
        save_dict = {
            'c': self.c,
            'counts': self.counts,
            'rewards': self.rewards,
            'total_trials': self.total_trials
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"✓ Bandit model saved to {path}")
    
    def load(self, path: str):
        """Load bandit state"""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.c = save_dict['c']
        self.counts = save_dict['counts']
        self.rewards = save_dict['rewards']
        self.total_trials = save_dict['total_trials']
        
        print(f"✓ Bandit model loaded from {path}")
        print(f"  Total trials: {self.total_trials}")


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("="*70)
    print("REINFORCEMENT LEARNING-BASED PROTOCOL SELECTION")
    print("="*70)
    
    # Example 1: Q-Learning
    print("\n" + "="*70)
    print("1. Q-LEARNING APPROACH")
    print("="*70)
    
    q_agent = QLearningProtocolSelector(alpha=0.1, gamma=0.95, epsilon=0.3)
    
    # Simulate training episodes
    print("\nTraining Q-Learning agent with simulated episodes...")
    
    training_scenarios = [
        # Low bandwidth, constrained -> MQTT performs best
        {
            'conditions': {
                'network': {'latency': 100, 'bandwidth': 2, 'packet_loss': 1},
                'resources': {'cpu_usage': 80, 'memory_usage': 85, 'battery_level': 30},
                'model': {'model_size_mb': 10},
                'mobility': {'is_mobile': False}
            },
            'best_protocol': 'mqtt',
            'performance': {'convergence_time': 45, 'final_accuracy': 0.85, 'avg_cpu_usage': 60}
        },
        # High performance -> gRPC performs best
        {
            'conditions': {
                'network': {'latency': 5, 'bandwidth': 100, 'packet_loss': 0},
                'resources': {'cpu_usage': 20, 'memory_usage': 30, 'battery_level': 100},
                'model': {'model_size_mb': 100},
                'mobility': {'is_mobile': False}
            },
            'best_protocol': 'grpc',
            'performance': {'convergence_time': 25, 'final_accuracy': 0.92, 'avg_cpu_usage': 40}
        },
        # Mobile -> QUIC performs best
        {
            'conditions': {
                'network': {'latency': 80, 'bandwidth': 5, 'packet_loss': 2},
                'resources': {'cpu_usage': 50, 'memory_usage': 60, 'battery_level': 45},
                'model': {'model_size_mb': 30},
                'mobility': {'is_mobile': True}
            },
            'best_protocol': 'quic',
            'performance': {'convergence_time': 38, 'final_accuracy': 0.88, 'avg_cpu_usage': 50}
        }
    ]
    
    # Train for 200 episodes
    for episode in range(200):
        scenario = random.choice(training_scenarios)
        
        # Agent selects protocol (with exploration)
        state = q_agent.get_state(scenario['conditions'])
        selected_protocol = q_agent.select_action(state, mode='train')
        
        # Simulate performance (best protocol gets best performance)
        if selected_protocol == scenario['best_protocol']:
            performance = scenario['performance']
        else:
            # Worse performance for suboptimal protocol
            performance = {
                'convergence_time': scenario['performance']['convergence_time'] * 1.5,
                'final_accuracy': scenario['performance']['final_accuracy'] * 0.9,
                'avg_cpu_usage': scenario['performance']['avg_cpu_usage'] * 1.2
            }
        
        # Train agent
        q_agent.train_episode(scenario['conditions'], selected_protocol, performance)
        
        if (episode + 1) % 50 == 0:
            stats = q_agent.get_statistics()
            print(f"  Episode {episode + 1}: Avg Reward={stats['avg_reward_last_100']:.2f}, ε={stats['epsilon']:.3f}")
    
    print(f"\n✓ Training completed!")
    stats = q_agent.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Total episodes: {stats['episodes']}")
    print(f"  Average reward (last 100): {stats['avg_reward_last_100']:.2f}")
    print(f"  Q-table size: {stats['q_table_size']} states")
    print(f"  Action distribution: {stats['action_distribution']}")
    
    # Test trained agent
    print("\nTesting trained agent on new conditions...")
    test_conditions = {
        'network': {'latency': 90, 'bandwidth': 3, 'packet_loss': 1.5},
        'resources': {'cpu_usage': 70, 'memory_usage': 75, 'battery_level': 35},
        'model': {'model_size_mb': 20},
        'mobility': {'is_mobile': False}
    }
    
    protocol, q_values = q_agent.select_best_protocol(test_conditions, verbose=True)
    
    # Save model
    q_agent.save("models/q_learning_protocol_selector.pkl")
    
    # Example 2: Multi-Armed Bandit
    print("\n" + "="*70)
    print("2. MULTI-ARMED BANDIT APPROACH")
    print("="*70)
    
    bandit = MultiArmedBanditSelector(c=2.0)
    
    print("\nTraining bandit with simulated trials...")
    
    # Simulate 100 trials with different protocol performances
    protocol_true_rewards = {'mqtt': 60, 'amqp': 55, 'grpc': 70, 'quic': 65, 'dds': 58}
    
    for trial in range(100):
        # Bandit selects protocol
        selected = bandit.select_protocol(verbose=False)
        
        # Observe reward (with noise)
        true_reward = protocol_true_rewards[selected]
        observed_reward = true_reward + np.random.normal(0, 5)
        
        # Update bandit
        bandit.update(selected, observed_reward)
        
        if (trial + 1) % 25 == 0:
            stats = bandit.get_statistics()
            print(f"  Trial {trial + 1}: Total trials={stats['total_trials']}")
    
    print(f"\n✓ Training completed!")
    
    # Final selection
    print("\nFinal protocol selection:")
    protocol = bandit.select_protocol(verbose=True)
    
    stats = bandit.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Total trials: {stats['total_trials']}")
    print(f"\nProtocol Performance:")
    for proto, data in sorted(stats['protocols'].items(), key=lambda x: x[1]['avg_reward'], reverse=True):
        print(f"  {proto:6s}: Avg Reward={data['avg_reward']:6.2f}, Trials={data['trials']:3d}")
    
    # Save model
    bandit.save("models/bandit_protocol_selector.pkl")
    
    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)
