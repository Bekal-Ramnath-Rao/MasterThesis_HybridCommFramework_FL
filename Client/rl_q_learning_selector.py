"""
Q-Learning Based Protocol Selector for Federated Learning

This module implements a Q-learning algorithm to dynamically select
the best communication protocol based on network conditions and 
system resources. RL logic is designed to run on CPU only (no TensorFlow
ops); client-side model training uses GPU separately.

Actions: MQTT, AMQP, gRPC, QUIC, DDS
Rewards (success, max 30): base 10, communication ≤10, resource ≥-5, battery ≥-5
Environment: Network conditions, Resources, Model size, Mobility
"""

import numpy as np
import json
import os
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pickle
import re
import shutil
import tempfile

try:
    import fcntl  # type: ignore
    _HAS_FCNTL = True
except ImportError:
    fcntl = None  # type: ignore
    _HAS_FCNTL = False


# --- Reward budget (success): base 10 + comm 10 + resource ≥ -5 + battery ≥ -5 → max 30 ---
# Communication time: log spread (seconds). Tuned from q_learning backups (uplink p50≈50s, downlink≈2s):
# T_HI 220 flattened the 10–90s band; ~120 improves slope for slow uploads without crushing totals.
_COMM_T_LO = 1.5
_COMM_T_HI = 120.0
_COMM_TIME_REWARD_MAX = 10.0
_RESOURCE_PENALTY_CAP = 5.0   # max magnitude (negative) from CPU/memory/bandwidth term
_BATTERY_PENALTY_CAP = 5.0    # max magnitude (negative) from SoC/energy term
# Resource penalties (pre-clip): CPU+memory and bandwidth (delta rate, not cumulative bytes).
_RESOURCE_CPU_MEM_WEIGHT = 6.0
_RESOURCE_BW_WEIGHT = 4.5
# Battery: soft energy norm (Joules). Backups often had SoC≈1 so drain term vanished; add direct
# energy_soft penalty so the [-5,0] band reflects protocol/round energy differences.
_ENERGY_SOFT_TAU_J = 22.0
_BATTERY_DRAIN_WEIGHT = 12.0   # (1 - SoC) * energy_soft
_BATTERY_SOC_WEIGHT = 1.8      # (1 - SoC) even when energy_soft is small
_BATTERY_ENERGY_WEIGHT = 3.0   # energy_soft always (uses more of the -5 cap vs SoC-only)
# Bandwidth: recent interface rate m/(m+k) with k = half-saturation Mbps (delta-based, not lifetime totals).
_BW_RATE_HALF_SAT_Mbps = 14.0


def _reward_communication_time(communication_time: float) -> float:
    """Higher reward for shorter times; log-scale spreads sensitivity in ~3–180 s range."""
    t = max(float(communication_time), 1e-6)
    denom = max(np.log1p(_COMM_T_HI) - np.log1p(_COMM_T_LO), 1e-9)
    u = (np.log1p(t) - np.log1p(_COMM_T_LO)) / denom
    u = float(np.clip(u, 0.0, 1.0))
    return _COMM_TIME_REWARD_MAX * (1.0 - u)


def _soft_energy_norm(energy_j: float, tau_j: float = _ENERGY_SOFT_TAU_J) -> float:
    """Maps Joules to (0,1) with diminishing returns; typical round energy stays in mid range."""
    e = max(float(energy_j), 0.0)
    tau = max(float(tau_j), 1e-6)
    return float(1.0 - np.exp(-e / tau))


def _bandwidth_norm_from_mbps(mbps: float) -> float:
    """Soft saturation of recent traffic rate (higher = busier link in the sampling window)."""
    m = max(float(mbps), 0.0)
    k = max(_BW_RATE_HALF_SAT_Mbps, 1e-6)
    return float(m / (m + k))


def _safe_scenario_filename_tag(scenario: Optional[str]) -> str:
    """Filesystem-safe fragment for archive filenames."""
    s = (scenario or "unknown").strip().lower()
    if not s:
        s = "unknown"
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = s.strip("_") or "unknown"
    return s[:80]


class QLearningProtocolSelector:
    """
    Q-Learning agent for selecting optimal communication protocol
    """
    
    # Protocol actions
    PROTOCOLS = ['mqtt', 'amqp', 'grpc', 'http3', 'dds']
    
    # Environment state dimensions
    NETWORK_CONDITIONS = ['excellent', 'moderate', 'poor']
    RESOURCE_LEVELS = ['high', 'low']
    MODEL_SIZES = ['small', 'medium', 'large']
    MOBILITY_LEVELS = ['static', 'mobile']
    
    def __init__(
        self,
        learning_rate: float = 0.12,
        discount_factor: float = 0.8,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.96,
        epsilon_min: float = 0.01,
        save_path: str = "q_table.pkl",
        initial_load_path: Optional[str] = None,
        use_communication_model_reward: bool = True,
    ):
        """
        Initialize Q-Learning Protocol Selector
        
        Args:
            learning_rate: Learning rate (alpha) for Q-learning
            discount_factor: Discount factor (gamma) for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            save_path: Path to save/load Q-table (persistent location recommended)
            initial_load_path: Optional path to load past experience first (e.g. pretrained .pkl).
                              If set and file exists, this is tried before save_path.
            use_communication_model_reward: When True, apply the communication-model
                                            T_calc penalty to the RL reward.
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.save_path = save_path
        self.initial_load_path = initial_load_path
        self.use_communication_model_reward = use_communication_model_reward
        
        # Q-table first axis = named network scenarios (simulation / experiment), extendable at runtime.
        # Defaults match legacy coarse buckets; new keys (e.g. very_poor, good) are appended via ensure_scenario().
        self.scenario_order: List[str] = list(self.NETWORK_CONDITIONS)
        self._scenario_to_idx: Dict[str, int] = {
            s: i for i, s in enumerate(self.scenario_order)
        }
        state_space_size = (
            len(self.scenario_order),
            len(self.RESOURCE_LEVELS),
            len(self.MODEL_SIZES),
            len(self.MOBILITY_LEVELS),
        )
        self.q_table = np.zeros(state_space_size + (len(self.PROTOCOLS),))
        
        # Statistics tracking
        self.episode_count = 0
        self.total_rewards = []
        self.protocol_usage = {p: 0 for p in self.PROTOCOLS}
        self.protocol_success = {p: 0 for p in self.PROTOCOLS}
        self.protocol_failures = {p: 0 for p in self.PROTOCOLS}

        # Track last scenario for epsilon reset detection
        self.last_scenario = None
        
        # Load existing Q-table if available (will reset if dimensions don't match)
        self.load_q_table()
        
        # History for learning
        self.state_history = []
        self.action_history = []
        self.reward_history = []

        # Q-value convergence tracking (for "end on Q convergence" mode)
        self._q_deltas = []  # abs(new_q - current_q) per update
        self._q_delta_window = 50  # keep last N deltas

        # Last reward breakdown (exact values for dashboard/logging)
        self._last_reward_breakdown = {}

    def _scenario_key_from_state(self, state: Dict) -> str:
        """Prefer explicit experiment/simulation scenario; else coarse measured ``network`` bucket."""
        raw = state.get("network_scenario")
        if raw is None:
            raw = state.get("scenario")
        if raw is not None and str(raw).strip():
            return str(raw).strip().lower()
        net = state.get("network", "moderate")
        return str(net).strip().lower()

    def _ensure_scenario_key(self, key: str) -> int:
        """Allocate a Q-table slice for this normalized scenario key if missing."""
        k = str(key).strip().lower() if key else "moderate"
        if not k:
            k = "moderate"
        if k not in self._scenario_to_idx:
            r = len(self.RESOURCE_LEVELS)
            m = len(self.MODEL_SIZES)
            mob = len(self.MOBILITY_LEVELS)
            p = len(self.PROTOCOLS)
            new_slice = np.zeros((1, r, m, mob, p))
            self.q_table = np.concatenate([self.q_table, new_slice], axis=0)
            idx = len(self.scenario_order)
            self.scenario_order.append(k)
            self._scenario_to_idx[k] = idx
            print(
                f"[Q-Learning] Extended Q-table for network scenario '{k}' "
                f"(slice {idx}, total={len(self.scenario_order)})"
            )
        return self._scenario_to_idx[k]

    def ensure_scenario(self, scenario_name: Optional[str]) -> int:
        """
        Ensure a row exists for this scenario name; grow the Q-table if needed.
        Call when the experiment selects a new network scenario so inference uses a distinct state slice.
        """
        if scenario_name is None or not str(scenario_name).strip():
            return self._ensure_scenario_key("moderate")
        return self._ensure_scenario_key(str(scenario_name))

    def get_state_index(self, state: Dict) -> Tuple[int, int, int, int]:
        """
        Convert state dictionary to indices for Q-table
        
        Args:
            state: Dictionary with optional network_scenario/scenario, network, resource, model_size, mobility
            
        Returns:
            Tuple of indices for Q-table access
        """
        scenario_idx = self._ensure_scenario_key(self._scenario_key_from_state(state))
        resource_idx = self.RESOURCE_LEVELS.index(state.get('resource', 'high'))
        model_idx = self.MODEL_SIZES.index(state.get('model_size', 'medium'))
        mobility_idx = self.MOBILITY_LEVELS.index(state.get('mobility', 'static'))
        
        return (scenario_idx, resource_idx, model_idx, mobility_idx)
    
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
            # Exploit: best known action (uses learned Q-table)
            # When training=False, this always uses the learned Q-table for inference
            action_idx = np.argmax(self.q_table[state_idx])
        
        protocol = self.PROTOCOLS[action_idx]
        self.protocol_usage[protocol] += 1
        
        # Store for learning
        self.state_history.append(state_idx)
        self.action_history.append(action_idx)
        
        return protocol

    def pop_last_selection(self) -> bool:
        """Remove the most recent (state, action) pair from learning history (e.g. after replacing an unavailable action)."""
        if not self.state_history or not self.action_history:
            return False
        self.state_history.pop()
        action_idx = self.action_history.pop()
        protocol = self.PROTOCOLS[action_idx]
        self.protocol_usage[protocol] = max(0, int(self.protocol_usage.get(protocol, 0)) - 1)
        return True

    def register_selection(self, state: Dict, protocol: str) -> str:
        """
        Record a fixed (state, protocol) pair for the next update_q_value, without epsilon-greedy sampling.
        Used when the actual transport is known (e.g. gRPC fallback) or to align history with delivery.
        """
        p = (protocol or "grpc").strip().lower()
        if p not in self.PROTOCOLS:
            p = "grpc"
        state_idx = self.get_state_index(state)
        action_idx = self.PROTOCOLS.index(p)
        self.protocol_usage[p] = int(self.protocol_usage.get(p, 0)) + 1
        self.state_history.append(state_idx)
        self.action_history.append(action_idx)
        return p
    
    def calculate_reward(
        self,
        communication_time: float,
        success: bool,
        resource_consumption: Dict[str, float],
        t_calc: Optional[float] = None,
        use_communication_model_reward: Optional[bool] = None,
    ) -> float:
        """
        Calculate reward based on multiple metrics.
        When t_calc (analytical transfer time from communication model) is provided,
        it reduces only the **communication** component (same 0–10 slot as wall-clock time).

        Args:
            communication_time: One-way communication time for this agent's link (seconds):
                uplink agent → client→server (model upload + metrics, etc.); downlink agent → server→client.
            success: Whether communication was successful
            resource_consumption: Dict with cpu, memory, bandwidth usage
            t_calc: Optional analytical transfer time (seconds). If set, high T_calc
                    reduces reward; low T_calc reduces reward less (communication model impact).
            use_communication_model_reward: Optional per-call override for whether
                                            T_calc should affect the reward.

        Returns:
            Calculated reward value
        """
        # Exact reward-component tracking for dashboard visibility
        cpu_usage = float(resource_consumption.get('cpu', 0.5))
        memory_usage = float(resource_consumption.get('memory', 0.5))
        bandwidth_usage = float(resource_consumption.get('bandwidth', 0.5))
        battery_level = float(resource_consumption.get('battery', 1.0))
        energy_j_raw = float(resource_consumption.get('energy_j', 0.0))
        energy_soft_metric = _soft_energy_norm(energy_j_raw)
        bw_mbps_est = resource_consumption.get('bandwidth_mbps_est')

        # Base reward for successful communication
        if not success:
            self._last_reward_breakdown = {
                'communication_time': float(communication_time),
                'success': False,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'bandwidth_usage': bandwidth_usage,
                'battery_level': battery_level,
                'energy_usage': float(resource_consumption.get('energy', energy_soft_metric)),
                'energy_j': energy_j_raw,
                'bandwidth_mbps_est': float(bw_mbps_est) if bw_mbps_est is not None else None,
                't_calc': float(t_calc) if t_calc is not None else None,
                'reward_base': -10.0,
                'reward_communication_time': 0.0,
                'reward_resource_penalty': 0.0,
                'reward_battery_penalty': 0.0,
                'reward_t_calc_penalty': 0.0,
                'reward_total': -10.0,
            }
            return -10.0  # Large penalty for failure

        # Budget: reward_base 10 + communication ≤10 + resource ≥-5 + battery ≥-5 (max 30)
        reward_base = 10.0
        reward = reward_base

        # Communication (wall time), max 10; optional T_calc (model) subtracts from same bucket only
        time_reward_raw = _reward_communication_time(communication_time)
        if use_communication_model_reward is None:
            use_communication_model_reward = self.use_communication_model_reward
        t_calc_penalty = 0.0
        if use_communication_model_reward and t_calc is not None and t_calc >= 0:
            t_calc_penalty = float(_COMM_TIME_REWARD_MAX * np.clip(
                (np.log1p(float(t_calc)) - np.log1p(_COMM_T_LO))
                / max(np.log1p(_COMM_T_HI) - np.log1p(_COMM_T_LO), 1e-9),
                0.0,
                1.0,
            ))
        time_reward = float(np.clip(time_reward_raw - t_calc_penalty, 0.0, _COMM_TIME_REWARD_MAX))
        reward += time_reward

        # Resource penalty (CPU+memory + bandwidth), clipped to [-_RESOURCE_PENALTY_CAP, 0]
        cpu_mem = 0.5 * (cpu_usage + memory_usage)
        resource_penalty_raw = -_RESOURCE_CPU_MEM_WEIGHT * cpu_mem - _RESOURCE_BW_WEIGHT * bandwidth_usage
        resource_penalty = float(np.clip(resource_penalty_raw, -_RESOURCE_PENALTY_CAP, 0.0))
        reward += resource_penalty

        # Battery / energy penalty, clipped to [-_BATTERY_PENALTY_CAP, 0]
        energy_soft = _soft_energy_norm(energy_j_raw)
        soc_stress = max(0.0, min(1.0, 1.0 - battery_level))
        battery_penalty_raw = (
            -_BATTERY_DRAIN_WEIGHT * soc_stress * energy_soft
            - _BATTERY_SOC_WEIGHT * soc_stress
            - _BATTERY_ENERGY_WEIGHT * energy_soft
        )
        low_batt_penalty = float(np.clip(battery_penalty_raw, -_BATTERY_PENALTY_CAP, 0.0))
        reward += low_batt_penalty

        self._last_reward_breakdown = {
            'communication_time': float(communication_time),
            'success': bool(success),
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'bandwidth_usage': bandwidth_usage,
            'battery_level': battery_level,
            'energy_usage': float(energy_soft),
            'energy_j': float(energy_j_raw),
            'bandwidth_mbps_est': float(bw_mbps_est) if bw_mbps_est is not None else None,
            't_calc': float(t_calc) if t_calc is not None else None,
            'reward_base': float(reward_base),
            'reward_communication_time': float(time_reward),
            'reward_communication_time_pre_t_calc': float(time_reward_raw),
            'reward_resource_penalty': float(resource_penalty),
            'reward_battery_penalty': float(low_batt_penalty),
            'reward_t_calc_penalty': -float(t_calc_penalty),
            'reward_total': float(reward),
        }

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
        q_delta = abs(new_q - current_q)
        self.q_table[state_idx][action_idx] = new_q

        # Track Q-deltas for convergence check
        self._q_deltas.append(q_delta)
        if len(self._q_deltas) > self._q_delta_window:
            self._q_deltas.pop(0)
        
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
    
    def reset_epsilon(self, experiment_id: str = None, scenario: str = None):
        """Reset epsilon to initial value (1.0) for re-exploration
        
        Args:
            experiment_id: Unique identifier for this experiment (prevents duplicate resets)
            scenario: Network scenario name for tracking
        
        Note: This resets ONLY epsilon, not the Q-table, allowing multi-scenario training
              to accumulate learning across all network conditions in one Q-table.
        """
        old_epsilon = self.epsilon
        self.epsilon = 1.0
        if experiment_id:
            self.last_experiment_id = experiment_id
        if scenario:
            self.last_scenario = scenario
            self.ensure_scenario(scenario)
        print(f"[Q-Learning] Epsilon reset from {old_epsilon:.4f} to {self.epsilon:.4f} for re-exploration")
        if experiment_id:
            print(f"[Q-Learning] Tracking experiment ID: {experiment_id}")
        if scenario:
            print(f"[Q-Learning] Tracking scenario: {scenario}")

    def _archive_canonical_pickle(self, scenario: Optional[str]) -> Optional[str]:
        """
        Copy save_path to same directory with timestamp + scenario in the name.
        Returns path to the archive file, or None if nothing was copied.
        """
        if not self.save_path or not os.path.isfile(self.save_path):
            return None
        d = os.path.dirname(os.path.abspath(self.save_path)) or "."
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            pass
        base = os.path.basename(self.save_path)
        stem, ext = os.path.splitext(base)
        if not ext:
            ext = ".pkl"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = _safe_scenario_filename_tag(scenario)
        archive_name = f"{stem}_archive_{ts}_{tag}{ext}"
        archive_path = os.path.join(d, archive_name)
        try:
            shutil.copy2(self.save_path, archive_path)
            print(f"[Q-Learning] Archived Q-table snapshot → {archive_path}")
            return archive_path
        except Exception as e:
            print(f"[Q-Learning] Warning: could not archive {self.save_path}: {e}")
            return None

    def begin_fresh_training_for_scenario(
        self,
        scenario: Optional[str],
        experiment_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Start an isolated training run for one network scenario while keeping a **single**
        canonical pickle (``save_path``) for inference across scenarios.

        1. If ``save_path`` exists, copy it to ``<stem>_archive_<timestamp>_<scenario>.pkl``.
        2. Zero only this scenario's Q-table slice; other scenario rows are unchanged.
        3. Reset epsilon, episode/history counters, and protocol stats (same run bookkeeping).

        Use this when switching training experiments so argmax for this scenario does not
        reuse Q-values from a previous run on the same scenario name.
        """
        archive_path = self._archive_canonical_pickle(scenario)
        skey = (
            str(scenario).strip().lower()
            if scenario and str(scenario).strip()
            else "moderate"
        )
        si = self.ensure_scenario(skey)
        self.q_table[si, :, :, :, :] = 0.0

        old_epsilon = self.epsilon
        self.epsilon = 1.0
        if experiment_id:
            self.last_experiment_id = experiment_id
        self.last_scenario = str(scenario).strip() if scenario and str(scenario).strip() else None

        self.episode_count = 0
        self.total_rewards = []
        self.protocol_usage = {p: 0 for p in self.PROTOCOLS}
        self.protocol_success = {p: 0 for p in self.PROTOCOLS}
        self.protocol_failures = {p: 0 for p in self.PROTOCOLS}
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self._q_deltas = []
        self._last_reward_breakdown = {}

        print(
            f"[Q-Learning] Fresh training for scenario '{skey}' (slice {si}): "
            f"Q-slice zeroed, epsilon {old_epsilon:.4f} → 1.0; other scenario slices unchanged"
        )
        return archive_path
    
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

    def _merge_env_overlay_scenarios(self) -> List[str]:
        """
        Scenario names this process may overwrite when merging into a shared on-disk Q-table.
        Set RL_QTABLE_MERGE=1 and either RL_QTABLE_OVERLAY_SCENARIOS=excellent,moderate or rely on
        last_scenario (e.g. from reset_epsilon / experiment flag) so parallel runs only touch
        their slice and do not zero out rows trained by other processes.
        """
        raw = os.getenv("RL_QTABLE_OVERLAY_SCENARIOS", "").strip()
        if raw:
            return [s.strip().lower() for s in raw.split(",") if s.strip()]
        merge_on = os.getenv("RL_QTABLE_MERGE", "").lower() in ("1", "true", "yes")
        if merge_on:
            ls = getattr(self, "last_scenario", None)
            if ls is not None and str(ls).strip():
                return [str(ls).strip().lower()]
        return []

    def _read_q_table_pickle(self, path: str) -> Optional[Dict]:
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[Q-Learning] Warning: could not read Q-table for merge at {path}: {e}")
            return None

    def _tail_shape(self) -> Tuple[int, int, int, int]:
        return (
            len(self.RESOURCE_LEVELS),
            len(self.MODEL_SIZES),
            len(self.MOBILITY_LEVELS),
            len(self.PROTOCOLS),
        )

    def _merge_disk_and_self_payload(self, disk: Optional[Dict], overlay: List[str]) -> Dict:
        """Build pickle dict: disk baseline + this process's rows for overlay / new scenarios."""
        tail = self._tail_shape()
        self_data = {
            "q_table": self.q_table,
            "scenario_order": list(self.scenario_order),
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
            "total_rewards": self.total_rewards,
            "protocol_usage": dict(self.protocol_usage),
            "protocol_success": dict(self.protocol_success),
            "protocol_failures": dict(self.protocol_failures),
            "last_scenario": getattr(self, "last_scenario", None),
        }
        if disk is None:
            return self_data

        dq = disk.get("q_table")
        if dq is None or dq.ndim != 5 or dq.shape[1:] != tail:
            print("[Q-Learning] Merge skipped: disk Q-table shape mismatch; writing this process only")
            return self_data

        n_disk = int(dq.shape[0])
        saved_order = disk.get("scenario_order")
        if saved_order is not None and len(saved_order) == n_disk:
            disk_order = [str(s).strip().lower() for s in saved_order]
        elif n_disk == len(self.NETWORK_CONDITIONS):
            disk_order = list(self.NETWORK_CONDITIONS)
        else:
            print("[Q-Learning] Merge skipped: cannot map disk scenario_order; writing this process only")
            return self_data

        disk_set = set(disk_order)
        merged_order: List[str] = []
        seen: set = set()
        for s in disk_order:
            if s not in seen:
                merged_order.append(s)
                seen.add(s)
        for s in self.scenario_order:
            s = str(s).strip().lower()
            if s not in seen:
                merged_order.append(s)
                seen.add(s)

        r, m, mob, p = tail
        merged_q = np.zeros((len(merged_order), r, m, mob, p))
        merged_idx = {s: i for i, s in enumerate(merged_order)}

        for i, s in enumerate(disk_order):
            j = merged_idx[s]
            merged_q[j] = np.array(dq[i], copy=True)

        for s in self.scenario_order:
            s = str(s).strip().lower()
            if s not in disk_set and s in self._scenario_to_idx:
                j = merged_idx[s]
                merged_q[j] = np.array(self.q_table[self._scenario_to_idx[s]], copy=True)

        overlay_set = {str(x).strip().lower() for x in overlay if str(x).strip()}
        for s in overlay_set:
            if s in self._scenario_to_idx:
                j = merged_idx[s]
                merged_q[j] = np.array(self.q_table[self._scenario_to_idx[s]], copy=True)

        def _sum_maps(a: Dict, b: Dict) -> Dict:
            out = {k: int(v) for k, v in (a or {}).items()}
            for k, v in (b or {}).items():
                out[k] = out.get(k, 0) + int(v)
            return out

        disk_ep = int(disk.get("episode_count", 0) or 0)
        ep_merged = max(disk_ep, int(self.episode_count))
        tr_disk = disk.get("total_rewards") or []
        tr_self = self.total_rewards or []
        total_rewards_merged = list(tr_self) if self.episode_count >= disk_ep else list(tr_disk)

        merged = {
            "q_table": merged_q,
            "scenario_order": merged_order,
            "epsilon": self.epsilon if self.episode_count >= disk_ep else disk.get("epsilon", self.epsilon),
            "episode_count": ep_merged,
            "total_rewards": total_rewards_merged,
            "protocol_usage": _sum_maps(disk.get("protocol_usage"), self.protocol_usage),
            "protocol_success": _sum_maps(disk.get("protocol_success"), self.protocol_success),
            "protocol_failures": _sum_maps(disk.get("protocol_failures"), self.protocol_failures),
            "last_scenario": getattr(self, "last_scenario", None) or disk.get("last_scenario"),
        }
        return merged

    def _atomic_pickle_dump(self, path: str, data: Dict) -> None:
        d = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp = tempfile.mkstemp(prefix=".qtable_", suffix=".tmp", dir=d)
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(data, f)
            os.replace(tmp, path)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            raise

    def save_q_table(self):
        """
        Save Q-table and statistics to disk.

        Parallel training (several containers/processes, one network scenario each)::

            RL_QTABLE_MERGE=1
            RL_QTABLE_OVERLAY_SCENARIOS=excellent   # or moderate, poor, congested_light, ...

        Uses a file lock and merges this process's scenario slice(s) into the existing file so other
        scenarios' rows are preserved.         Without RL_QTABLE_MERGE, behavior is a single-process
        overwrite (unchanged).
        """
        overlay = self._merge_env_overlay_scenarios()
        try:
            if overlay and _HAS_FCNTL:
                lock_path = f"{self.save_path}.lock"
                with open(lock_path, "a+") as lock_f:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
                    try:
                        disk = self._read_q_table_pickle(self.save_path)
                        payload = self._merge_disk_and_self_payload(disk, overlay)
                        self._atomic_pickle_dump(self.save_path, payload)
                    finally:
                        fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                print(f"[Q-Learning] Saved Q-table (merge, overlay={overlay}) to {self.save_path}")
            elif overlay and not _HAS_FCNTL:
                disk = self._read_q_table_pickle(self.save_path)
                payload = self._merge_disk_and_self_payload(disk, overlay)
                self._atomic_pickle_dump(self.save_path, payload)
                print(
                    f"[Q-Learning] Saved Q-table (merge, no fcntl lock) to {self.save_path}"
                )
            else:
                data = {
                    "q_table": self.q_table,
                    "scenario_order": list(self.scenario_order),
                    "epsilon": self.epsilon,
                    "episode_count": self.episode_count,
                    "total_rewards": self.total_rewards,
                    "protocol_usage": self.protocol_usage,
                    "protocol_success": self.protocol_success,
                    "protocol_failures": self.protocol_failures,
                    "last_scenario": getattr(self, "last_scenario", None),
                }
                self._atomic_pickle_dump(self.save_path, data)
                print(f"[Q-Learning] Saved Q-table to {self.save_path}")
        except Exception as e:
            print(f"[Q-Learning] Error saving Q-table: {e}")
    
    def _try_load_from_path(self, path: str) -> bool:
        """Try to load Q-table from given path. Return True if loaded successfully."""
        if not path or not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            loaded_q_table = data.get('q_table')
            tail_shape = (
                len(self.RESOURCE_LEVELS),
                len(self.MODEL_SIZES),
                len(self.MOBILITY_LEVELS),
                len(self.PROTOCOLS),
            )
            if loaded_q_table is None or loaded_q_table.ndim != 5:
                return False
            if loaded_q_table.shape[1:] != tail_shape:
                print(
                    f"[Q-Learning] Q-table tail shape mismatch at {path}: "
                    f"expected (*,{tail_shape}), got {loaded_q_table.shape}"
                )
                return False
            n_scen = int(loaded_q_table.shape[0])
            saved_order = data.get('scenario_order')
            if saved_order is not None and len(saved_order) == n_scen:
                scenario_order = [str(s).strip().lower() for s in saved_order]
            elif n_scen == len(self.NETWORK_CONDITIONS):
                # Legacy pickle: first axis was coarse network buckets only
                scenario_order = list(self.NETWORK_CONDITIONS)
            else:
                print(
                    f"[Q-Learning] Q-table at {path}: cannot map axis 0 (size {n_scen}) "
                    f"to scenario names; expected scenario_order of length {n_scen}"
                )
                return False
            self.q_table = loaded_q_table
            self.scenario_order = scenario_order
            self._scenario_to_idx = {s: i for i, s in enumerate(self.scenario_order)}
            self.epsilon = data.get('epsilon', self.epsilon)
            self.episode_count = data.get('episode_count', 0)
            self.total_rewards = data.get('total_rewards', [])
            self.protocol_usage = data.get('protocol_usage', self.protocol_usage)
            self.protocol_success = data.get('protocol_success', self.protocol_success)
            self.protocol_failures = data.get('protocol_failures', self.protocol_failures)
            self.last_scenario = data.get('last_scenario', None)  # Load last scenario
            print(f"[Q-Learning] Loaded Q-table from {path} (past experience)")
            print(
                f"[Q-Learning] Episodes: {self.episode_count}, Epsilon: {self.epsilon:.4f}, "
                f"scenarios={len(self.scenario_order)}"
            )
            if self.last_scenario:
                print(f"[Q-Learning] Last scenario: {self.last_scenario}")
            return True
        except Exception as e:
            print(f"[Q-Learning] Error loading Q-table from {path}: {e}")
            return False

    def load_q_table(self):
        """Load Q-table from disk: try initial_load_path (past experience) first, then save_path."""
        # Try optional pretrained / past-experience path first
        if self.initial_load_path and self._try_load_from_path(self.initial_load_path):
            return
        # Then try default save path (e.g. shared_data or cwd)
        if self._try_load_from_path(self.save_path):
            return
        print(f"[Q-Learning] No existing Q-table found. Starting with fresh Q-table for {len(self.PROTOCOLS)} protocols: {self.PROTOCOLS}")
    
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
    
    def get_last_q_delta(self) -> float:
        """Return the last Q-update delta (for logging)."""
        return self._q_deltas[-1] if self._q_deltas else 0.0

    def get_last_q_data(self) -> Dict:
        """
        Return a dict of last Q-learning step data (for logging/display).
        Keys: q_delta, epsilon, episode_count, avg_reward_last_100, last_state, last_action, last_reward.
        """
        avg_reward = (
            float(np.mean(self.total_rewards[-100:]))
            if self.total_rewards else 0.0
        )
        data = {
            'q_delta': self.get_last_q_delta(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'avg_reward_last_100': avg_reward,
            'last_state': self.state_history[-1] if self.state_history else None,
            'last_action': self.action_history[-1] if self.action_history else None,
            'last_reward': self.reward_history[-1] if self.reward_history else None,
            'last_reward_breakdown': self.get_last_reward_breakdown(),
        }
        return data

    def get_last_reward_breakdown(self) -> Dict:
        """Return exact last reward inputs and per-component contributions."""
        return dict(self._last_reward_breakdown) if self._last_reward_breakdown else {}

    def check_q_converged(
        self,
        threshold: Optional[float] = None,
        patience: Optional[int] = None,
    ) -> bool:
        """
        RL agent convergence (when USE_QL_CONVERGENCE is enabled):

        1. Epsilon is below the exploitation cap (default strictly below 0.1).
        2. The last N protocol selections are identical (same action index),
           where N is `patience` if provided, else ``RL_CONVERGENCE_SAME_PROTOCOL_STREAK``
           (default 5).

        The ``threshold`` argument is kept for call-site compatibility; Q-delta
        convergence is no longer used.
        """
        _ = threshold  # legacy parameter; Q-delta criterion removed
        streak = (
            patience
            if patience is not None
            else int(os.getenv("RL_CONVERGENCE_SAME_PROTOCOL_STREAK", "5"))
        )
        epsilon_cap = float(os.getenv("RL_CONVERGENCE_EPSILON_CAP", "0.1"))
        if self.epsilon >= epsilon_cap:
            return False
        if len(self.action_history) < streak:
            return False
        tail = self.action_history[-streak:]
        first = tail[0]
        return all(a == first for a in tail)

    def reset_episode(self):
        """Reset episode history"""
        self.state_history = []
        self.action_history = []
    
    def reset_q_table(self):
        """
        Reset Q-table to initial state (all zeros) and reset statistics.
        This clears all learned knowledge and starts fresh.
        Useful when new protocols are added (e.g., HTTP/3) or starting new training.
        """
        self.scenario_order = list(self.NETWORK_CONDITIONS)
        self._scenario_to_idx = {s: i for i, s in enumerate(self.scenario_order)}
        state_space_size = (
            len(self.scenario_order),
            len(self.RESOURCE_LEVELS),
            len(self.MODEL_SIZES),
            len(self.MOBILITY_LEVELS),
        )
        self.q_table = np.zeros(state_space_size + (len(self.PROTOCOLS),))
        
        # Reset epsilon to initial value
        self.epsilon = 1.0
        
        # Reset statistics
        self.episode_count = 0
        self.total_rewards = []
        self.protocol_usage = {p: 0 for p in self.PROTOCOLS}
        self.protocol_success = {p: 0 for p in self.PROTOCOLS}
        self.protocol_failures = {p: 0 for p in self.PROTOCOLS}
        
        # Reset history
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self._q_deltas = []
        self._last_reward_breakdown = {}
        
        # Delete saved Q-table file if it exists
        if os.path.exists(self.save_path):
            try:
                os.remove(self.save_path)
                print(f"[Q-Learning] Deleted existing Q-table file: {self.save_path}")
            except Exception as e:
                print(f"[Q-Learning] Warning: Could not delete Q-table file {self.save_path}: {e}")
        
        print(f"[Q-Learning] Q-table reset complete. Starting fresh with {len(self.PROTOCOLS)} protocols: {self.PROTOCOLS}")


# Legacy constant (some docs/scripts); reward uses _soft_energy_norm / _ENERGY_SOFT_TAU_J instead.
E_REF = 10.0

# Reference payload size for normalizing "communication data footprint" in the memory term
PAYLOAD_BYTES_REF = 12 * 1024 * 1024

# Normalized protocol stack / library footprint (0..1) for RL memory — heavier stacks cost more
PROTOCOL_MEMORY_STACK_NORM = {
    'mqtt': 0.08,
    'amqp': 0.12,
    'grpc': 0.15,
    'quic': 0.18,
    'http3': 0.20,
    'dds': 0.14,
}


class EnvironmentStateManager:
    """
    Manages environment state for RL agent
    Tracks network conditions, resources, model size, and mobility
    """
    
    def __init__(self):
        self.current_state = {
            'network': 'moderate',
            'resource': 'high',
            'model_size': 'medium',
            'mobility': 'static',
            # When set, RL Q-table indexes this simulation scenario (good, very_poor, ...); else use ``network``.
            'network_scenario': None,
        }
        
        # Resource monitoring
        self.cpu_usage_history = []
        self.memory_usage_history = []
        self.bandwidth_usage_history = []

        # Battery / energy state (continuous; not part of Q-table state space)
        self.battery_soc = 1.0  # state of charge [0, 1]
        self.last_energy_j = 0.0  # Joules used in the last round
        # Last psutil net_io snapshot for delta-rate bandwidth (avoids cumulative saturation)
        self._net_io_prev: Optional[Tuple[int, int, float]] = None

    def update_battery(self, soc: float, last_energy_j: float) -> None:
        """Update battery state of charge [0,1] and last-round energy in Joules."""
        self.battery_soc = max(0.0, min(1.0, soc))
        self.last_energy_j = max(0.0, last_energy_j)

    def update_network_condition(self, condition: str):
        """Update network condition"""
        if condition in QLearningProtocolSelector.NETWORK_CONDITIONS:
            self.current_state['network'] = condition

    def update_network_scenario(self, scenario: Optional[str]) -> None:
        """Set explicit simulation/experiment scenario for RL state (None = use measured ``network`` only)."""
        if scenario is None or not str(scenario).strip():
            self.current_state['network_scenario'] = None
        else:
            self.current_state['network_scenario'] = str(scenario).strip().lower()
    
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
        if latency_ms < 20 and bandwidth_mbps > 50:
            return 'excellent'
        elif latency_ms < 50 and bandwidth_mbps > 10:
            return 'moderate'
        elif latency_ms < 120 and bandwidth_mbps > 1:
            return 'poor'
        else:
            return 'poor'
    
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
        else:
            return 'low'
    
    def get_current_state(self) -> Dict:
        """Get current environment state"""
        return self.current_state.copy()
    
    def get_resource_consumption(
        self,
        protocol: Optional[str] = None,
        uplink_payload_bytes: Optional[int] = None,
        downlink_payload_bytes: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Get current resource consumption metrics.

        When ``protocol`` and payload sizes are provided, the memory term blends (1) live RSS
        pressure from psutil, (2) protocol stack weight, and (3) normalized uplink+downlink
        payload bytes for this round so heavier protocols and larger FL transfers increase the
        memory penalty in ``calculate_reward``.

        Returns:
            Dictionary with normalized (0-1) resource usage, battery, soft energy norm,
            raw ``energy_j`` (Joules), and ``bandwidth_mbps_est`` when available.
        """
        try:
            import psutil

            cpu = psutil.cpu_percent(interval=0.1) / 100.0
            memory_psutil = psutil.virtual_memory().percent / 100.0

            p = (protocol or "").strip().lower()
            stack_norm = PROTOCOL_MEMORY_STACK_NORM.get(p, 0.12)
            ul = float(uplink_payload_bytes or 0)
            dl = float(downlink_payload_bytes or 0)
            data_norm = min(1.0, (ul + dl) / float(PAYLOAD_BYTES_REF))
            memory = min(
                1.0,
                0.40 * memory_psutil + 0.35 * stack_norm + 0.25 * data_norm,
            )

            net_io = psutil.net_io_counters()
            now = time.time()
            sent, recv = int(net_io.bytes_sent), int(net_io.bytes_recv)
            if self._net_io_prev is None:
                self._net_io_prev = (sent, recv, now)
                mbps = 0.0
                bandwidth = 0.38  # neutral until a second sample establishes a rate
            else:
                ps, pr, pt = self._net_io_prev
                dt = max(now - pt, 0.05)
                d_bytes = max(0, (sent - ps) + (recv - pr))
                mbps = (d_bytes * 8.0) / (dt * 1e6)
                self._net_io_prev = (sent, recv, now)
                bandwidth = _bandwidth_norm_from_mbps(mbps)

            energy_j = float(max(0.0, self.last_energy_j))
            energy_soft = _soft_energy_norm(energy_j)

            return {
                'cpu': cpu,
                'memory': memory,
                'bandwidth': bandwidth,
                'bandwidth_mbps_est': float(mbps),
                'battery': self.battery_soc,
                'energy': energy_soft,
                'energy_j': energy_j,
            }
        except Exception:
            ej = float(max(0.0, getattr(self, 'last_energy_j', 0.0)))
            return {
                'cpu': 0.5,
                'memory': 0.5,
                'bandwidth': 0.45,
                'bandwidth_mbps_est': None,
                'battery': getattr(self, 'battery_soc', 1.0),
                'energy': _soft_energy_norm(ej),
                'energy_j': ej,
            }


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
        resources = env_manager.get_resource_consumption()
        
        reward = selector.calculate_reward(
            comm_time, success, resources
        )
        print(f"Reward: {reward:.2f}")
        
        # Update Q-value
        selector.update_q_value(reward, done=True)
        selector.end_episode()
    
    # Print final statistics
    selector.print_statistics()
