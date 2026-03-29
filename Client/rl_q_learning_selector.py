"""
Q-Learning Based Protocol Selector for Federated Learning

This module implements a Q-learning algorithm to dynamically select
the best communication protocol based on network conditions and 
system resources. RL logic is designed to run on CPU only (no TensorFlow
ops); client-side model training uses GPU separately.

Actions: MQTT, AMQP, gRPC, QUIC, DDS
Rewards (success, max 30): base 10, communication ≤10, resource ≥-5, battery ≥-5
Environment (Q-table state): communication-time bucket, resource availability, battery SoC bucket
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

# region agent log
_AGENT_DEBUG_LOG_PATH = "/home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL/.cursor/debug-c387e1.log"


def _agent_debug_log(
    location: str,
    message: str,
    data: dict,
    hypothesis_id: str = "",
    run_id: str = "pre-fix",
) -> None:
    try:
        line = json.dumps(
            {
                "sessionId": "c387e1",
                "runId": run_id,
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000),
            },
            default=str,
        )
        with open(_AGENT_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# endregion

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
    
    # Q-table state: communication time bucket, resource availability, battery SoC bucket
    COMM_LEVELS = ["low", "mid", "high"]  # wall-clock comm time: low=fast, high=slow
    RESOURCE_LEVELS = ["high", "low"]  # high = plenty of CPU/memory; low = heavily loaded
    BATTERY_LEVELS = ["high", "low"]  # high SoC vs low SoC
    
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
        
        # Data-driven comm-time bucket boundaries (seconds); tune after Phase-1 collection
        self.comm_t_low = 30.0
        self.comm_t_high = 90.0
        # Mean CPU+memory load (0–100): <= threshold → resource "high"; separate per uplink/downlink agent
        self.resource_load_threshold = 30.0
        # SoC in [0,1]: >= threshold → battery "high"; separate per uplink/downlink agent
        self.battery_soc_threshold = 0.35

        # Training/inference metadata only: not used in Q-state indexing (comm/resource/battery).
        self.data_network_scenario: Optional[str] = None

        self.q_table = np.zeros(
            (
                len(self.COMM_LEVELS),
                len(self.RESOURCE_LEVELS),
                len(self.BATTERY_LEVELS),
                len(self.PROTOCOLS),
            )
        )
        
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

        # Q-value convergence tracking (for "end on Q convergence" mode)
        self._q_deltas = []  # abs(new_q - current_q) per update
        self._q_delta_window = 50  # keep last N deltas

        # Last reward breakdown (exact values for dashboard/logging)
        self._last_reward_breakdown = {}

    def comm_time_to_level(self, t: float) -> str:
        """Map communication time (seconds) to a discrete bucket; thresholds are tunable after data collection."""
        t = float(t)
        if t <= self.comm_t_low:
            return "low"
        if t <= self.comm_t_high:
            return "mid"
        return "high"

    def set_data_network_scenario(self, scenario: Optional[str]) -> None:
        """Persisted with the Q-table; for inference/logging only — not part of RL state."""
        if scenario is None or (isinstance(scenario, str) and not str(scenario).strip()):
            self.data_network_scenario = None
        else:
            self.data_network_scenario = str(scenario).strip().lower()

    def get_data_network_scenario(self) -> Optional[str]:
        """Scenario tag stored with this agent (same Q-table file); not used in ``get_state_index``."""
        return self.data_network_scenario

    def ensure_scenario(self, scenario_name: Optional[str]) -> None:
        """Backward compatibility: no scenario axis on the Q-table; store tag for pickle + inference."""
        self.set_data_network_scenario(scenario_name)

    def get_state_index(self, state: Dict) -> Tuple[int, int, int]:
        """
        Convert state dictionary to indices for Q-table.

        Expected keys: comm_level, resource, battery_level (each a string label).
        """
        comm = state.get("comm_level", "mid")
        if comm not in self.COMM_LEVELS:
            comm = "mid"
        res = state.get("resource", "high")
        if res not in self.RESOURCE_LEVELS:
            res = "high"
        batt = state.get("battery_level", "high")
        if batt not in self.BATTERY_LEVELS:
            batt = "high"
        try:
            c_idx = self.COMM_LEVELS.index(comm)
            r_idx = self.RESOURCE_LEVELS.index(res)
            b_idx = self.BATTERY_LEVELS.index(batt)
        except ValueError as e:
            _agent_debug_log(
                "rl_q_learning_selector.py:get_state_index",
                "invalid_discrete_state_key",
                {
                    "error": str(e),
                    "comm_level": state.get("comm_level"),
                    "resource": state.get("resource"),
                    "battery_level": state.get("battery_level"),
                    "allowed_comm": list(self.COMM_LEVELS),
                    "allowed_resource": list(self.RESOURCE_LEVELS),
                    "allowed_battery": list(self.BATTERY_LEVELS),
                },
                hypothesis_id="H1",
            )
            raise

        _agent_debug_log(
            "rl_q_learning_selector.py:get_state_index",
            "state_to_q_indices",
            {
                "comm_level": comm,
                "resource": res,
                "battery_level": batt,
                "idx": [int(c_idx), int(r_idx), int(b_idx)],
            },
            hypothesis_id="H5",
        )
        return (c_idx, r_idx, b_idx)

    def _state_tuple_to_dict(self, state_idx: Tuple[int, int, int]) -> Dict[str, str]:
        c_idx, r_idx, b_idx = state_idx
        return {
            "comm_level": self.COMM_LEVELS[c_idx],
            "resource": self.RESOURCE_LEVELS[r_idx],
            "battery_level": self.BATTERY_LEVELS[b_idx],
        }
    
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
            explored = True
        else:
            # Exploit: best known action (uses learned Q-table)
            # When training=False, this always uses the learned Q-table for inference
            action_idx = np.argmax(self.q_table[state_idx])
            explored = False
        
        # region agent log
        _agent_debug_log(
            "rl_q_learning_selector.py:select_protocol",
            "protocol_choice",
            {
                "training": bool(training),
                "epsilon": float(self.epsilon),
                "explored": explored,
                "action_idx": int(action_idx),
                "protocol": self.PROTOCOLS[int(action_idx)],
                "max_q": float(np.max(self.q_table[state_idx])),
            },
            hypothesis_id="H4",
        )
        # endregion

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
            # region agent log
            _agent_debug_log(
                "rl_q_learning_selector.py:update_q_value",
                "skipped_empty_history",
                {
                    "state_hist_len": len(self.state_history),
                    "action_hist_len": len(self.action_history),
                    "reward": float(reward),
                },
                hypothesis_id="H2",
            )
            # endregion
            return
        
        state_idx = self.state_history[-1]
        action_idx = self.action_history[-1]
        
        current_q = self.q_table[state_idx + (action_idx,)]
        
        # Calculate new Q-value
        max_next_q = None
        if done or next_state is None:
            # Terminal state (no bootstrap from next state)
            new_q = current_q + self.learning_rate * (reward - current_q)
            if done:
                _boot_reason = "done=True"
            else:
                _boot_reason = "next_state=None"
        else:
            # Non-terminal state: use Bellman equation (r + γ max_a' Q(s',a'))
            next_state_idx = self.get_state_index(next_state)
            max_next_q = float(np.max(self.q_table[next_state_idx]))
            new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * max_next_q - current_q
            )
            _boot_reason = "next_state provided"
        if os.environ.get("RL_Q_LOG_BOOTSTRAP", "1").strip().lower() not in ("0", "false", "no"):
            print(
                f"[Q-Learning] Q-update bootstrap: "
                f"discounted_future={'yes' if max_next_q is not None else 'no'} "
                f"(reason={_boot_reason}"
                + (
                    f", gamma={self.discount_factor:.4f}, max_Q(s')={max_next_q:.4f}, "
                    f"gamma*max_Q(s')={self.discount_factor * max_next_q:.4f}"
                    if max_next_q is not None
                    else ""
                )
                + ")"
            )

        # Update Q-table
        q_delta = abs(new_q - current_q)
        self.q_table[state_idx + (action_idx,)] = new_q

        # region agent log
        _agent_debug_log(
            "rl_q_learning_selector.py:update_q_value",
            "q_updated",
            {
                "reward": float(reward),
                "current_q": float(current_q),
                "new_q": float(new_q),
                "q_delta": float(q_delta),
                "action_idx": int(action_idx),
                "protocol": self.PROTOCOLS[int(action_idx)],
                "terminal": bool(done or next_state is None),
            },
            hypothesis_id="H2",
        )
        # endregion

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
        """Reset epsilon to initial value (1.0) for re-exploration.

        Args:
            experiment_id: Unique identifier for this experiment (prevents duplicate resets)
            scenario: Ignored (legacy); Q-table no longer has a scenario axis.
        """
        _ = scenario
        old_epsilon = self.epsilon
        self.epsilon = 1.0
        if experiment_id:
            self.last_experiment_id = experiment_id
        print(f"[Q-Learning] Epsilon reset from {old_epsilon:.4f} to {self.epsilon:.4f} for re-exploration")
        if experiment_id:
            print(f"[Q-Learning] Tracking experiment ID: {experiment_id}")

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

    def begin_fresh_training(
        self,
        experiment_id: Optional[str] = None,
        scenario: Optional[str] = None,
    ) -> Optional[str]:
        """
        Archive current save_path if present, zero the full Q-table, reset epsilon and stats.

        ``scenario`` is only used as an archive filename tag (legacy API compatibility).
        """
        archive_path = self._archive_canonical_pickle(scenario)
        self.q_table.fill(0.0)

        old_epsilon = self.epsilon
        self.epsilon = 1.0
        if experiment_id:
            self.last_experiment_id = experiment_id

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
            f"[Q-Learning] Fresh training: Q-table zeroed, epsilon {old_epsilon:.4f} → 1.0"
        )
        return archive_path

    def begin_fresh_training_for_scenario(
        self,
        scenario: Optional[str],
        experiment_id: Optional[str] = None,
    ) -> Optional[str]:
        """Legacy alias: full-table reset; ``scenario`` is used only for archive naming."""
        return self.begin_fresh_training(experiment_id=experiment_id, scenario=scenario)
    
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
        """When RL_QTABLE_MERGE=1, merge this process's Q-table with on-disk file (element-wise max)."""
        merge_on = os.getenv("RL_QTABLE_MERGE", "").lower() in ("1", "true", "yes")
        return ["merge"] if merge_on else []

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
            len(self.COMM_LEVELS),
            len(self.RESOURCE_LEVELS),
            len(self.BATTERY_LEVELS),
            len(self.PROTOCOLS),
        )

    def _merge_disk_and_self_payload(self, disk: Optional[Dict], overlay: List[str]) -> Dict:
        """Merge on-disk Q-table with this process (element-wise max) when overlay merge is active."""
        self_data = {
            "q_table": self.q_table,
            "comm_t_low": self.comm_t_low,
            "comm_t_high": self.comm_t_high,
            "resource_load_threshold": self.resource_load_threshold,
            "battery_soc_threshold": self.battery_soc_threshold,
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
            "total_rewards": self.total_rewards,
            "protocol_usage": dict(self.protocol_usage),
            "protocol_success": dict(self.protocol_success),
            "protocol_failures": dict(self.protocol_failures),
            "data_network_scenario": self.data_network_scenario,
        }
        if disk is None or not overlay:
            return self_data

        dq = disk.get("q_table")
        tail = self._tail_shape()
        if dq is None or dq.ndim != 4 or tuple(dq.shape) != tail:
            print("[Q-Learning] Merge skipped: disk Q-table shape mismatch; writing this process only")
            return self_data

        merged_q = np.maximum(np.asarray(dq, dtype=float), np.asarray(self.q_table, dtype=float))

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

        dns_self = self.data_network_scenario
        dns_disk = disk.get("data_network_scenario") if isinstance(disk, dict) else None
        merged_dns = dns_self if dns_self is not None else dns_disk

        return {
            "q_table": merged_q,
            "comm_t_low": float(disk.get("comm_t_low", self.comm_t_low)),
            "comm_t_high": float(disk.get("comm_t_high", self.comm_t_high)),
            "resource_load_threshold": float(
                disk.get("resource_load_threshold", self.resource_load_threshold)
            ),
            "battery_soc_threshold": float(disk.get("battery_soc_threshold", self.battery_soc_threshold)),
            "epsilon": self.epsilon if self.episode_count >= disk_ep else disk.get("epsilon", self.epsilon),
            "episode_count": ep_merged,
            "total_rewards": total_rewards_merged,
            "protocol_usage": _sum_maps(disk.get("protocol_usage"), self.protocol_usage),
            "protocol_success": _sum_maps(disk.get("protocol_success"), self.protocol_success),
            "protocol_failures": _sum_maps(disk.get("protocol_failures"), self.protocol_failures),
            "data_network_scenario": merged_dns,
        }

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

        Parallel training: set ``RL_QTABLE_MERGE=1`` to merge element-wise max with the existing file (same shape).
        Otherwise overwrites the file.
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
                print(f"[Q-Learning] Saved Q-table (merge) to {self.save_path}")
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
                    "comm_t_low": self.comm_t_low,
                    "comm_t_high": self.comm_t_high,
                    "resource_load_threshold": self.resource_load_threshold,
                    "battery_soc_threshold": self.battery_soc_threshold,
                    "epsilon": self.epsilon,
                    "episode_count": self.episode_count,
                    "total_rewards": self.total_rewards,
                    "protocol_usage": self.protocol_usage,
                    "protocol_success": self.protocol_success,
                    "protocol_failures": self.protocol_failures,
                    "data_network_scenario": self.data_network_scenario,
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
            tail_shape = self._tail_shape()
            if loaded_q_table is None:
                return False
            if loaded_q_table.ndim == 4 and tuple(loaded_q_table.shape) == tail_shape:
                self.q_table = np.asarray(loaded_q_table, dtype=float)
            else:
                print(
                    f"[Q-Learning] Q-table shape mismatch at {path}: "
                    f"expected {tail_shape}, got {getattr(loaded_q_table, 'shape', None)} "
                    "(legacy 5D scenario pickles are not auto-migrated)"
                )
                return False
            self.comm_t_low = float(data.get('comm_t_low', self.comm_t_low))
            self.comm_t_high = float(data.get('comm_t_high', self.comm_t_high))
            self.resource_load_threshold = float(
                data.get('resource_load_threshold', self.resource_load_threshold)
            )
            self.battery_soc_threshold = float(
                data.get('battery_soc_threshold', self.battery_soc_threshold)
            )
            self.epsilon = data.get('epsilon', self.epsilon)
            self.episode_count = data.get('episode_count', 0)
            self.total_rewards = data.get('total_rewards', [])
            self.protocol_usage = data.get('protocol_usage', self.protocol_usage)
            self.protocol_success = data.get('protocol_success', self.protocol_success)
            self.protocol_failures = data.get('protocol_failures', self.protocol_failures)
            self.data_network_scenario = data.get("data_network_scenario")
            if self.data_network_scenario is not None:
                self.data_network_scenario = str(self.data_network_scenario).strip().lower() or None
            print(f"[Q-Learning] Loaded Q-table from {path} (past experience)")
            print(
                f"[Q-Learning] Episodes: {self.episode_count}, Epsilon: {self.epsilon:.4f}, "
                f"shape={tuple(self.q_table.shape)}"
            )
            if self.data_network_scenario:
                print(
                    f"[Q-Learning] data_network_scenario (metadata, not RL state): "
                    f"{self.data_network_scenario!r}"
                )
            return True
        except Exception as e:
            print(f"[Q-Learning] Error loading Q-table from {path}: {e}")
            return False

    def load_q_table(self):
        """Load Q-table from disk: try initial_load_path (past experience) first, then save_path."""
        # Try optional pretrained / past-experience path first
        if self.initial_load_path and self._try_load_from_path(self.initial_load_path):
            # region agent log
            _agent_debug_log(
                "rl_q_learning_selector.py:load_q_table",
                "q_table_loaded",
                {
                    "source": "initial_load_path",
                    "path": self.initial_load_path,
                    "q_shape": list(self.q_table.shape),
                    "epsilon": float(self.epsilon),
                },
                hypothesis_id="H3",
            )
            # endregion
            return
        # Then try default save path (e.g. shared_data or cwd)
        if self._try_load_from_path(self.save_path):
            # region agent log
            _agent_debug_log(
                "rl_q_learning_selector.py:load_q_table",
                "q_table_loaded",
                {
                    "source": "save_path",
                    "path": self.save_path,
                    "q_shape": list(self.q_table.shape),
                    "epsilon": float(self.epsilon),
                },
                hypothesis_id="H3",
            )
            # endregion
            return
        # region agent log
        _agent_debug_log(
            "rl_q_learning_selector.py:load_q_table",
            "q_table_fresh",
            {
                "source": "none",
                "initial_load_path": self.initial_load_path,
                "save_path": self.save_path,
                "q_shape": list(self.q_table.shape),
            },
            hypothesis_id="H3",
        )
        # endregion
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
        last_state_idx = self.state_history[-1] if self.state_history else None
        last_state_dict = (
            self._state_tuple_to_dict(last_state_idx)
            if last_state_idx is not None
            else None
        )
        data = {
            'q_delta': self.get_last_q_delta(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'avg_reward_last_100': avg_reward,
            'last_state': last_state_dict,
            'last_state_idx': last_state_idx,
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

        The last N protocol selections are identical (same action index), where N
        is `patience` if provided, else ``RL_CONVERGENCE_SAME_PROTOCOL_STREAK``
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
        self.q_table = np.zeros(
            (
                len(self.COMM_LEVELS),
                len(self.RESOURCE_LEVELS),
                len(self.BATTERY_LEVELS),
                len(self.PROTOCOLS),
            )
        )
        
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
        self.data_network_scenario = None
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
    Tracks RL state for the Q-learning agent: comm time bucket, resource level, battery bucket.
    Continuous SoC and comm thresholds feed the discrete buckets.
    """
    
    def __init__(self):
        self.current_state = {
            "comm_level": "mid",
            "resource": "high",
            "battery_level": "high",
        }
        # Tunable alongside QLearningProtocolSelector.comm_t_low / comm_t_high after data collection
        self.comm_t_low = 30.0
        self.comm_t_high = 90.0
        self.battery_soc_threshold = 0.35  # SoC >= threshold -> battery_level "high"
        # Mean CPU+memory load (0–100): values <= threshold -> resource "high" (headroom); else "low"
        self.resource_load_threshold = 30.0
        
        # Resource monitoring
        self.cpu_usage_history = []
        self.memory_usage_history = []
        self.bandwidth_usage_history = []

        # Battery / energy state (continuous)
        self.battery_soc = 1.0  # state of charge [0, 1]
        self.last_energy_j = 0.0  # Joules used in the last round
        # Last psutil net_io snapshot for delta-rate bandwidth (avoids cumulative saturation)
        self._net_io_prev: Optional[Tuple[int, int, float]] = None
        # Mirrors Q-table metadata; not used in discrete Q-state (comm/resource/battery).
        self.data_network_scenario: Optional[str] = None

    def comm_time_to_level(self, t: float) -> str:
        """Map wall-clock communication time (seconds) to a discrete comm bucket."""
        t = float(t)
        if t <= self.comm_t_low:
            return "low"
        if t <= self.comm_t_high:
            return "mid"
        return "high"

    def battery_soc_to_level(self, soc: Optional[float] = None) -> str:
        """Map SoC in [0,1] to battery_level for Q-table indexing."""
        s = float(self.battery_soc if soc is None else soc)
        s = max(0.0, min(1.0, s))
        return "high" if s >= self.battery_soc_threshold else "low"

    def update_comm_level_from_time(
        self,
        communication_time: float,
        comm_t_low: Optional[float] = None,
        comm_t_high: Optional[float] = None,
    ) -> None:
        """Set ``current_state['comm_level']`` from measured communication time.

        When ``comm_t_low`` / ``comm_t_high`` are omitted, uses ``self.comm_t_low`` /
        ``self.comm_t_high`` (typically aligned with the uplink selector). Pass explicit
        thresholds when mapping **downlink** wall times using the downlink agent's bounds.
        """
        lo = float(self.comm_t_low if comm_t_low is None else comm_t_low)
        hi = float(self.comm_t_high if comm_t_high is None else comm_t_high)
        t = float(communication_time)
        if t <= lo:
            self.current_state["comm_level"] = "low"
        elif t <= hi:
            self.current_state["comm_level"] = "mid"
        else:
            self.current_state["comm_level"] = "high"

    def sync_battery_level_from_soc(self, soc: Optional[float] = None) -> None:
        """Refresh ``current_state['battery_level']`` from SoC."""
        self.current_state["battery_level"] = self.battery_soc_to_level(soc)

    def sync_comm_thresholds_from_selector(self, selector: "QLearningProtocolSelector") -> None:
        """Keep env comm + resource/battery thresholds aligned with the uplink selector."""
        self.comm_t_low = float(selector.comm_t_low)
        self.comm_t_high = float(selector.comm_t_high)
        self.resource_load_threshold = float(selector.resource_load_threshold)
        self.battery_soc_threshold = float(selector.battery_soc_threshold)

    def resource_level_for_threshold(
        self, cpu_percent: float, memory_percent: float, threshold: float
    ) -> str:
        """``high`` = headroom (mean load <= threshold), ``low`` = stressed."""
        avg_usage = (float(cpu_percent) + float(memory_percent)) / 2.0
        return "high" if avg_usage <= float(threshold) else "low"

    def battery_level_for_threshold(self, soc: float, threshold: float) -> str:
        s = max(0.0, min(1.0, float(soc)))
        return "high" if s >= float(threshold) else "low"

    def state_for_rl_selector(
        self,
        selector: "QLearningProtocolSelector",
        comm_time: float,
        cpu_percent: float,
        memory_percent: float,
    ) -> Dict[str, str]:
        """Discrete state for uplink vs downlink Q-agent (each selector carries its own thresholds)."""
        comm = selector.comm_time_to_level(float(comm_time))
        res = self.resource_level_for_threshold(
            cpu_percent, memory_percent, selector.resource_load_threshold
        )
        batt = self.battery_level_for_threshold(self.battery_soc, selector.battery_soc_threshold)
        return {"comm_level": comm, "resource": res, "battery_level": batt}

    def neutral_rl_state_before_boundaries(self) -> Dict[str, str]:
        """Discrete tuple for ε-random protocol picks during Phase 1 only.

        Not derived from measured comm / load / battery. After all collection rounds,
        :func:`compute_state_boundaries_from_minmax` derives thresholds from the stored
        samples (min–max span); then :func:`state_for_rl_selector` is used with real cuts.
        """
        return {"comm_level": "mid", "resource": "high", "battery_level": "high"}

    def update_battery(self, soc: float, last_energy_j: float) -> None:
        """Update battery state of charge [0,1] and last-round energy in Joules."""
        self.battery_soc = max(0.0, min(1.0, soc))
        self.last_energy_j = max(0.0, last_energy_j)
        self.sync_battery_level_from_soc(self.battery_soc)

    def update_network_condition(self, condition: str) -> None:
        """Legacy: no longer part of Q-table state (optional logging only)."""
        _ = condition

    def update_network_scenario(
        self,
        scenario: Optional[str],
        *,
        rl_uplink: Optional["QLearningProtocolSelector"] = None,
        rl_downlink: Optional["QLearningProtocolSelector"] = None,
    ) -> None:
        """Store scenario tag for inference/logging; optional sync to RL agents for pickle persistence.

        ``comm_level`` / ``resource`` / ``battery_level`` remain the only RL state axes.
        """
        if scenario is None or (isinstance(scenario, str) and not str(scenario).strip()):
            self.data_network_scenario = None
        else:
            self.data_network_scenario = str(scenario).strip().lower()
        if rl_uplink is not None:
            rl_uplink.set_data_network_scenario(self.data_network_scenario)
        if rl_downlink is not None:
            rl_downlink.set_data_network_scenario(self.data_network_scenario)

    def get_data_network_scenario(self) -> Optional[str]:
        """Scenario tag last set via :meth:`update_network_scenario` (env mirror of selector metadata)."""
        return self.data_network_scenario

    def update_resource_level(self, level: str):
        """Update resource availability: 'high' or 'low'."""
        if level in QLearningProtocolSelector.RESOURCE_LEVELS:
            self.current_state["resource"] = level

    def update_model_size(self, size: str) -> None:
        """Legacy: model size is not used in Q-table indexing."""
        _ = size

    def update_mobility(self, mobility: str) -> None:
        """Legacy: mobility is not used in Q-table indexing."""
        _ = mobility
    
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
        thr = float(getattr(self, "resource_load_threshold", 30.0))
        if avg_usage <= thr:
            return "high"
        return "low"
    
    def get_current_state(self) -> Dict:
        """Discrete RL state plus optional ``data_network_scenario`` (ignored by Q-state indexing)."""
        out = self.current_state.copy()
        out["data_network_scenario"] = self.data_network_scenario
        return out
    
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


def compute_comm_terciles_from_samples(comm_times: List[float]) -> Optional[Tuple[float, float]]:
    """Upper edges of low/mid comm buckets from min–max range (33% / 66% of span)."""
    if not comm_times:
        return None
    ct = [float(x) for x in comm_times]
    cmin, cmax = min(ct), max(ct)
    cspan = max(cmax - cmin, 1e-9)
    return (float(cmin + 0.33 * cspan), float(cmin + 0.66 * cspan))


def _load_or_soc_split_threshold(samples: List[float]) -> float:
    """Two-state split at min + 33% of observed range (same rule as comm/resource/battery)."""
    if not samples:
        return 30.0
    vs = [float(x) for x in samples]
    vmin, vmax = min(vs), max(vs)
    span = max(vmax - vmin, 1e-9)
    return float(vmin + 0.33 * span)


def compute_state_boundaries_from_minmax(
    uplink_comm_times: List[float],
    resource_loads_uplink: List[float],
    battery_socs_uplink: List[float],
    downlink_comm_times: Optional[List[float]] = None,
    resource_loads_downlink: Optional[List[float]] = None,
    battery_socs_downlink: Optional[List[float]] = None,
) -> Optional[Dict[str, float]]:
    """
    Run **after** all Phase-1 rounds: uses **min/max** of the collected samples (per series),
    then derives cut points (comm: 33%/66% along the span; resource/battery: split at min+33% of span).

    Separate uplink vs downlink for comm, resource, and battery where samples exist.
    """
    if not uplink_comm_times or not resource_loads_uplink or not battery_socs_uplink:
        return None
    uc = compute_comm_terciles_from_samples(uplink_comm_times)
    if uc is None:
        return None
    comm_t_low, comm_t_high = uc
    dl = downlink_comm_times if downlink_comm_times else []
    dc = compute_comm_terciles_from_samples(dl) if len(dl) >= 1 else None
    if dc is not None:
        comm_t_low_dl, comm_t_high_dl = dc
    else:
        comm_t_low_dl, comm_t_high_dl = comm_t_low, comm_t_high

    r_ul = _load_or_soc_split_threshold(resource_loads_uplink)
    b_ul = _load_or_soc_split_threshold(battery_socs_uplink)
    rdl = resource_loads_downlink if resource_loads_downlink else []
    bdl = battery_socs_downlink if battery_socs_downlink else []
    r_dl = _load_or_soc_split_threshold(rdl) if len(rdl) >= 1 else r_ul
    b_dl = _load_or_soc_split_threshold(bdl) if len(bdl) >= 1 else b_ul

    return {
        "comm_t_low": float(comm_t_low),
        "comm_t_high": float(comm_t_high),
        "comm_t_low_dl": float(comm_t_low_dl),
        "comm_t_high_dl": float(comm_t_high_dl),
        "resource_load_threshold_ul": float(r_ul),
        "resource_load_threshold_dl": float(r_dl),
        "battery_soc_threshold_ul": float(b_ul),
        "battery_soc_threshold_dl": float(b_dl),
    }


def apply_state_boundaries_to_rl_selectors(
    uplink: QLearningProtocolSelector,
    downlink: QLearningProtocolSelector,
    env_manager: EnvironmentStateManager,
    bounds: Dict[str, float],
) -> None:
    uplink.comm_t_low = float(bounds["comm_t_low"])
    uplink.comm_t_high = float(bounds["comm_t_high"])
    uplink.resource_load_threshold = float(bounds["resource_load_threshold_ul"])
    uplink.battery_soc_threshold = float(bounds["battery_soc_threshold_ul"])
    downlink.comm_t_low = float(bounds.get("comm_t_low_dl", bounds["comm_t_low"]))
    downlink.comm_t_high = float(bounds.get("comm_t_high_dl", bounds["comm_t_high"]))
    downlink.resource_load_threshold = float(bounds["resource_load_threshold_dl"])
    downlink.battery_soc_threshold = float(bounds["battery_soc_threshold_dl"])
    # Env mirrors uplink agent (shared logging / legacy paths)
    env_manager.comm_t_low = float(bounds["comm_t_low"])
    env_manager.comm_t_high = float(bounds["comm_t_high"])
    env_manager.resource_load_threshold = float(bounds["resource_load_threshold_ul"])
    env_manager.battery_soc_threshold = float(bounds["battery_soc_threshold_ul"])


def finalize_rl_boundary_collection_and_start_training(
    uplink: QLearningProtocolSelector,
    downlink: QLearningProtocolSelector,
    env_manager: EnvironmentStateManager,
    uplink_comm_times: List[float],
    resource_loads_uplink: List[float],
    battery_socs_uplink: List[float],
    client_id: int = 0,
    downlink_comm_times: Optional[List[float]] = None,
    resource_loads_downlink: Optional[List[float]] = None,
    battery_socs_downlink: Optional[List[float]] = None,
) -> bool:
    """
    Phase 2: compute boundaries from collected samples, apply to env + both selectors,
    then reset Q-tables and epsilon for Phase 3 (RL training with decay).
    """
    bounds = compute_state_boundaries_from_minmax(
        uplink_comm_times,
        resource_loads_uplink,
        battery_socs_uplink,
        downlink_comm_times=downlink_comm_times,
        resource_loads_downlink=resource_loads_downlink,
        battery_socs_downlink=battery_socs_downlink,
    )
    if bounds is None:
        print("[Q-Learning] Boundary collection: no samples; keeping default thresholds.")
    else:
        apply_state_boundaries_to_rl_selectors(uplink, downlink, env_manager, bounds)
        print(
            "[Q-Learning] Phase 2 (boundaries from data): "
            f"uplink comm: {bounds['comm_t_low']:.4f}s, {bounds['comm_t_high']:.4f}s; "
            f"downlink comm: {bounds['comm_t_low_dl']:.4f}s, {bounds['comm_t_high_dl']:.4f}s; "
            f"resource thr % (ul/dl): {bounds['resource_load_threshold_ul']:.2f}, {bounds['resource_load_threshold_dl']:.2f}; "
            f"SoC thr (ul/dl): {bounds['battery_soc_threshold_ul']:.4f}, {bounds['battery_soc_threshold_dl']:.4f}"
        )
    uplink.begin_fresh_training(experiment_id=f"client{client_id}_uplink_rl_train")
    downlink.begin_fresh_training(experiment_id=f"client{client_id}_downlink_rl_train")
    return bounds is not None


def collect_comm_resource_battery_stats(
    selector: QLearningProtocolSelector,
    env_manager: EnvironmentStateManager,
    n_rounds: int = 20,
) -> Tuple[List[float], List[str], List[float]]:
    """
    **Offline / test helper only**: synthetic random walk (not real FL rounds).
    For production, the unified client collects metrics during actual FL rounds; see
    ``finalize_rl_boundary_collection_and_start_training``.

    Runs n_rounds with epsilon=1.0 (pure exploration) without Q-learning updates.
    """
    old_eps = selector.epsilon
    selector.epsilon = 1.0
    old_usage = dict(selector.protocol_usage)

    comm_times: List[float] = []
    resource_levels: List[str] = []
    battery_socs: List[float] = []

    for _ in range(n_rounds):
        state = env_manager.get_current_state()
        # Protocol choice is random due to epsilon=1.0
        _ = selector.select_protocol(state, training=True)

        # Placeholder: replace with one real FL round + measured communication_time
        communication_time = float(np.random.uniform(5.0, 150.0))
        comm_times.append(communication_time)
        env_manager.update_comm_level_from_time(communication_time)

        cpu_p = float(np.random.uniform(10, 90))
        mem_p = float(np.random.uniform(10, 90))
        res = env_manager.detect_resource_level(cpu_p, mem_p)
        env_manager.update_resource_level(res)
        resource_levels.append(res)

        soc = float(np.random.uniform(0.1, 1.0))
        env_manager.update_battery(soc, float(np.random.uniform(0.0, 40.0)))
        battery_socs.append(env_manager.battery_soc)

    selector.epsilon = old_eps
    selector.protocol_usage = old_usage
    selector.state_history.clear()
    selector.action_history.clear()
    selector.reward_history.clear()

    return comm_times, resource_levels, battery_socs


if __name__ == "__main__":
    print("Q-Learning Protocol Selector - Test")

    selector = QLearningProtocolSelector(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
    )
    env_manager = EnvironmentStateManager()
    env_manager.sync_comm_thresholds_from_selector(selector)

    print("\n--- Phase 1: bucket discovery (exploration only, no Q-updates) ---")
    ct, rl, bs = collect_comm_resource_battery_stats(selector, env_manager, n_rounds=10)
    print(
        f"Collected comm_times={len(ct)}, resource labels={len(rl)}, battery_socs={len(bs)}; "
        "threshold tuning: TODO (e.g. quantiles of comm_times)."
    )

    for episode in range(50):
        print(f"\nEpisode {episode + 1}")

        env_manager.current_state["comm_level"] = str(
            np.random.choice(QLearningProtocolSelector.COMM_LEVELS)
        )
        env_manager.update_resource_level(
            str(np.random.choice(QLearningProtocolSelector.RESOURCE_LEVELS))
        )
        env_manager.battery_soc = float(np.random.uniform(0.05, 1.0))
        env_manager.sync_battery_level_from_soc()

        state = env_manager.get_current_state()
        print(f"State: {state}")

        protocol = selector.select_protocol(state, training=True)
        print(f"Selected protocol: {protocol}")

        success = np.random.random() > 0.2
        comm_time = float(np.random.uniform(0.1, 120.0))
        resources = env_manager.get_resource_consumption(protocol=protocol)

        reward = selector.calculate_reward(comm_time, success, resources)
        print(f"Reward: {reward:.2f}")

        env_manager.update_comm_level_from_time(comm_time)

        next_state = env_manager.get_current_state()
        selector.update_q_value(reward, next_state=next_state, done=False)
        selector.end_episode()

    selector.print_statistics()
