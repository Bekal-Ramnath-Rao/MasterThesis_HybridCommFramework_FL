"""
Shared battery/energy model for FL clients (single-protocol and unified).
Matches the energy model used in the unified Emotion client for fair comparison.
"""
import time
from typing import Optional

# Energy / battery model constants (tunable, same as FL_Client_Unified Emotion)
k_tx = 1e-8
k_rx = 1e-8
E_fixed = 0.1  # J
P_CPU_MAX = 10.0  # W
BATTERY_CAP_J = 60.0 * 3600.0  # 60 Wh example capacity (Joules)
PROTOCOL_ENERGY_ALPHA = {
    "mqtt": 1.0,
    "amqp": 1.1,
    "grpc": 1.2,
    "quic": 1.1,
    "http3": 1.25,
    "dds": 1.1,
}
PROTOCOL_CPU_BETA = {
    "mqtt": 1.0,
    "amqp": 1.05,
    "grpc": 1.1,
    "quic": 1.05,
    "http3": 1.15,
    "dds": 1.0,
}


class BatteryModel:
    """
    Tracks battery state of charge (SoC) per round using the same energy model
    as the unified client: E_radio (tx/rx) + E_cpu (training + comm time).
    """

    def __init__(self, protocol: str = "mqtt"):
        self.protocol = protocol.lower()
        self.battery_soc = 1.0  # state of charge [0, 1]
        self.last_energy_j = 0.0  # Joules used in the last round
        self.cumulative_energy_j = 0.0  # Total Joules consumed so far

    def update(
        self,
        bytes_sent: int,
        bytes_recv: int,
        training_time_sec: float,
        communication_time_sec: float,
        cpu_util_percent: Optional[float] = None,
    ) -> float:
        """
        Update battery SoC after one FL round.
        Returns energy consumed this round in Joules.
        """
        try:
            import psutil
            cpu_util = cpu_util_percent if cpu_util_percent is not None else psutil.cpu_percent(interval=0.0)
        except Exception:
            cpu_util = 50.0

        bits_tx = bytes_sent * 8
        bits_rx = bytes_recv * 8
        t_round = training_time_sec + communication_time_sec
        alpha = PROTOCOL_ENERGY_ALPHA.get(self.protocol, 1.0)
        beta = PROTOCOL_CPU_BETA.get(self.protocol, 1.0)
        E_radio_baseline = k_tx * bits_tx + k_rx * bits_rx + E_fixed
        E_radio = alpha * E_radio_baseline
        E_cpu = P_CPU_MAX * (cpu_util / 100.0) * t_round * beta
        E_total = E_radio + E_cpu

        soc = self.battery_soc
        delta_soc = E_total / BATTERY_CAP_J
        new_soc = soc - delta_soc
        self.battery_soc = max(0.0, min(1.0, new_soc))
        self.last_energy_j = E_total
        self.cumulative_energy_j += E_total
        return E_total

    @property
    def consumption_fraction(self) -> float:
        """Fraction of battery consumed so far (0 = none, 1 = full)."""
        return 1.0 - self.battery_soc
