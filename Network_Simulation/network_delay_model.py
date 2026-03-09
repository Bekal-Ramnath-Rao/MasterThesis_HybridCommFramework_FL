#!/usr/bin/env python3
"""
Realistic network delay and jitter modeling using Gaussian distributions.
Parses latency/jitter strings from NETWORK_SCENARIOS and provides sampling
for one-way delay (optionally with extra jitter term) for use in normal and
diagnostic pipeline experiments.
"""

import re
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np


# Canonical network scenarios; latency and jitter are parsed to float ms for Gaussian models.
NETWORK_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "excellent": {
        "name": "Excellent Network (LAN)",
        "latency": "2ms",
        "jitter": "1ms",
        "bandwidth": "100mbit",
        "loss": "0.01%"
    },
    "good": {
        "name": "Good Network (Broadband)",
        "latency": "20ms",
        "jitter": "5ms",
        "bandwidth": "100mbit",
        "loss": "0.1%"
    },
    "moderate": {
        "name": "Moderate Network (4G/LTE)",
        "latency": "30ms",
        "jitter": "10ms",
        "bandwidth": "10mbit",
        "loss": "0.3%"
    },
    "poor": {
        "name": "Poor Network (3G)",
        "latency": "120ms",
        "jitter": "30ms",
        "bandwidth": "1mbit",
        "loss": "0.5%"
    },
    "very_poor": {
        "name": "Very Poor Network (Edge/2G)",
        "latency": "400ms",
        "jitter": "100ms",
        "bandwidth": "75kbit",
        "loss": "1%"
    },
    "satellite": {
        "name": "Satellite Network",
        "latency": "600ms",
        "jitter": "50ms",
        "bandwidth": "5mbit",
        "loss": "1.5%"
    },
    "congested_light": {
        "name": "Light Congestion (Shared Network)",
        "latency": "30ms",
        "jitter": "15ms",
        "bandwidth": "10mbit",
        "loss": "1.5%"
    },
    "congested_moderate": {
        "name": "Moderate Congestion (Peak Hours)",
        "latency": "75ms",
        "jitter": "35ms",
        "bandwidth": "5mbit",
        "loss": "3.5%"
    },
    "congested_heavy": {
        "name": "Heavy Congestion (Network Overload)",
        "latency": "150ms",
        "jitter": "60ms",
        "bandwidth": "2mbit",
        "loss": "6%"
    }
}


def _parse_delay_ms(value: str) -> float:
    """Parse a delay string (e.g. '20ms', '1.5ms') into float milliseconds."""
    if not value:
        return 0.0
    s = str(value).strip().lower()
    m = re.match(r"(\d+(?:\.\d+)?)\s*(\w*)", s)
    if not m:
        return 0.0
    val, unit = float(m.group(1)), (m.group(2) or "").lower()
    if "ms" in unit:
        return val
    # assume seconds if "s" or number only
    if "s" in unit or (unit and "m" not in unit):
        return val * 1000.0
    return val


@dataclass
class NetworkScenarioModel:
    """Gaussian model for one-way delay and jitter for a network scenario."""

    name: str
    mean_delay_ms: float
    sigma_delay_ms: float
    mean_jitter_ms: float
    sigma_jitter_ms: float


def build_models(
    network_scenarios: Dict[str, Dict[str, Any]],
    sigma_factor: float = 0.05,
) -> Dict[str, NetworkScenarioModel]:
    """
    Build delay/jitter models from a NETWORK_SCENARIOS-like dict.
    Parses latency and jitter strings to numeric ms; sigma = sigma_factor * mean.
    """
    models: Dict[str, NetworkScenarioModel] = {}
    for key, scenario in network_scenarios.items():
        latency_str = scenario.get("latency") or "0ms"
        jitter_str = scenario.get("jitter") or "0ms"
        mean_delay_ms = _parse_delay_ms(latency_str)
        mean_jitter_ms = _parse_delay_ms(jitter_str)
        sigma_delay_ms = sigma_factor * mean_delay_ms if mean_delay_ms > 0 else 0.0
        sigma_jitter_ms = sigma_factor * mean_jitter_ms if mean_jitter_ms > 0 else 0.0
        models[key] = NetworkScenarioModel(
            name=scenario.get("name", key),
            mean_delay_ms=mean_delay_ms,
            sigma_delay_ms=sigma_delay_ms,
            mean_jitter_ms=mean_jitter_ms,
            sigma_jitter_ms=sigma_jitter_ms,
        )
    return models


def sample_delay_ms(
    models: Dict[str, NetworkScenarioModel],
    scenario_key: str,
    use_extra_jitter: bool = False,
) -> float:
    """
    Sample one-way delay in milliseconds for the given scenario.

    (a) use_extra_jitter=False: single Gaussian only,
        delay ~ N(mean_delay_ms, sigma_delay_ms^2), clipped >= 0.
    (b) use_extra_jitter=True: base delay + 0-mean jitter,
        base ~ N(mean_delay_ms, sigma_delay_ms^2),
        jitter ~ N(0, sigma_jitter_ms^2),
        delay = base + jitter, clipped >= 0.
    """
    if scenario_key not in models:
        raise KeyError(f"Unknown scenario: {scenario_key}. Choose from: {list(models.keys())}")
    m = models[scenario_key]
    base = np.random.normal(m.mean_delay_ms, m.sigma_delay_ms)
    if use_extra_jitter and m.sigma_jitter_ms > 0:
        jitter = np.random.normal(0.0, m.sigma_jitter_ms)
        total = base + jitter
    else:
        total = base
    return float(max(0.0, total))


def sample_delay_and_jitter_ms(
    models: Dict[str, NetworkScenarioModel],
    scenario_key: str,
    use_extra_jitter: bool = False,
) -> tuple[float, float]:
    """
    Sample base delay and jitter in ms for tc netem (delay base_ms jitter_ms).
    Returns (base_delay_ms, jitter_ms); both are >= 0.

    - Delay: sampled from N(mean_delay_ms, sigma_delay_ms).
    - Jitter: always sampled from the scenario's jitter model when mean_jitter_ms > 0,
      i.e. N(mean_jitter_ms, sigma_jitter_ms) clipped to >= 0, so e.g. "excellent" (1ms)
      yields jitter around 1ms. When use_extra_jitter is True, the same distribution
      is used (no separate 0-mean term for jitter here; tc netem gets the sampled value).
    """
    if scenario_key not in models:
        raise KeyError(f"Unknown scenario: {scenario_key}. Choose from: {list(models.keys())}")
    m = models[scenario_key]
    base = np.random.normal(m.mean_delay_ms, m.sigma_delay_ms)
    base_ms = float(max(0.0, base))
    # Always use scenario jitter model for tc: Gaussian around mean_jitter_ms (e.g. 1ms for excellent)
    if m.mean_jitter_ms > 0:
        jitter = np.random.normal(m.mean_jitter_ms, m.sigma_jitter_ms if m.sigma_jitter_ms > 0 else 0.0)
        jitter_ms = float(max(0.0, jitter))
    else:
        jitter_ms = 0.0
    return (base_ms, jitter_ms)


# Alias for compatibility with diagnostic_pipeline (build_delay_models)
build_delay_models = build_models


if __name__ == "__main__":
    # Example: build models with sigma_factor=0.05, draw 10 samples for satellite and congested_heavy
    models = build_models(NETWORK_SCENARIOS, sigma_factor=0.05)
    print("sigma_factor = 0.05")
    print("10 samples — satellite (single Gaussian only):")
    for _ in range(10):
        d = sample_delay_ms(models, "satellite", use_extra_jitter=False)
        print(f"  {d:.2f} ms")
    print("10 samples — satellite (base Gaussian + extra jitter):")
    for _ in range(10):
        d = sample_delay_ms(models, "satellite", use_extra_jitter=True)
        print(f"  {d:.2f} ms")
    print("10 samples — congested_heavy (single Gaussian only):")
    for _ in range(10):
        d = sample_delay_ms(models, "congested_heavy", use_extra_jitter=False)
        print(f"  {d:.2f} ms")
    print("10 samples — congested_heavy (base Gaussian + extra jitter):")
    for _ in range(10):
        d = sample_delay_ms(models, "congested_heavy", use_extra_jitter=True)
        print(f"  {d:.2f} ms")
