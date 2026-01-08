"""
Dynamic Protocol Selection Engine for Federated Learning
Selects optimal communication protocol based on network, resource, and mobility conditions.
"""

import psutil
import time
import platform
import subprocess
import re
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class ProtocolScore:
    """Score details for a protocol"""
    protocol: str
    total_score: float
    network_score: float
    resource_score: float
    model_score: float
    mobility_score: float
    details: Dict[str, str]


class ProtocolSelector:
    """
    Decision engine for dynamic protocol selection in FL.
    
    Selection criteria:
    1. Network conditions (latency, bandwidth, packet loss, jitter)
    2. Resource availability (CPU, memory, battery)
    3. Model characteristics (size, update frequency)
    4. Mobility patterns (stationary vs mobile, handoffs)
    """
    
    def __init__(self, server_address: str = "localhost"):
        self.server_address = server_address
        
        # Weight factors for different criteria (sum = 1.0)
        self.weights = {
            "network": 0.40,      # 40% - Most important for communication
            "resources": 0.25,    # 25% - Device constraints
            "model": 0.20,        # 20% - Model characteristics
            "mobility": 0.15      # 15% - Movement patterns
        }
        
        # Protocol characteristics (0-100 scale, higher is better for each condition)
        self.protocol_profiles = {
            "mqtt": {
                "low_bandwidth": 95,      # Excellent for low bandwidth
                "high_latency": 80,       # Good tolerance for latency
                "packet_loss": 85,        # Good resilience
                "low_cpu": 95,            # Very light CPU usage
                "low_memory": 95,         # Very low memory footprint
                "large_model": 60,        # Okay for large models
                "high_mobility": 90,      # Excellent for mobile scenarios
                "battery_efficient": 95   # Most battery efficient
            },
            "amqp": {
                "low_bandwidth": 75,
                "high_latency": 75,
                "packet_loss": 80,
                "low_cpu": 80,
                "low_memory": 80,
                "large_model": 70,
                "high_mobility": 75,
                "battery_efficient": 80
            },
            "grpc": {
                "low_bandwidth": 70,
                "high_latency": 85,       # Good with HTTP/2
                "packet_loss": 65,
                "low_cpu": 60,            # Higher CPU usage
                "low_memory": 65,
                "large_model": 90,        # Excellent for large models
                "high_mobility": 50,      # Prefers stable connections
                "battery_efficient": 60
            },
            "quic": {
                "low_bandwidth": 80,
                "high_latency": 95,       # Excellent low latency
                "packet_loss": 95,        # Best packet loss handling
                "low_cpu": 70,
                "low_memory": 70,
                "large_model": 85,
                "high_mobility": 95,      # Excellent for mobility
                "battery_efficient": 70
            },
            "dds": {
                "low_bandwidth": 60,
                "high_latency": 90,       # Good real-time performance
                "packet_loss": 80,
                "low_cpu": 50,            # Higher overhead
                "low_memory": 55,
                "large_model": 75,
                "high_mobility": 70,
                "battery_efficient": 55
            }
        }
    
    def measure_network_conditions(self) -> Dict[str, float]:
        """Measure current network conditions"""
        conditions = {
            "latency": self._measure_latency(),
            "bandwidth": self._estimate_bandwidth(),
            "packet_loss": self._estimate_packet_loss(),
            "jitter": 0.0  # Can be measured over multiple pings
        }
        return conditions
    
    def measure_resource_availability(self) -> Dict[str, float]:
        """Measure current device resource availability"""
        resources = {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_available": psutil.virtual_memory().available / (1024**2),  # MB
            "memory_percent": psutil.virtual_memory().percent,
            "battery_level": self._get_battery_level(),
            "is_charging": self._is_charging()
        }
        return resources
    
    def assess_mobility(self) -> Dict[str, any]:
        """
        Assess mobility characteristics.
        In production: Use GPS, accelerometer, network handoff detection
        For simulation: Can be configured or estimated
        """
        # Placeholder - in real implementation, integrate with:
        # - GPS for velocity
        # - Network interface monitoring for handoffs
        # - Accelerometer for movement detection
        
        mobility = {
            "is_mobile": False,           # Default: stationary
            "velocity": 0.0,              # km/h
            "connection_stability": 100,  # 0-100, higher is more stable
            "handoffs_per_hour": 0        # Network handoffs
        }
        
        # Simple heuristic: check if on WiFi vs cellular
        try:
            if platform.system() == "Windows":
                # Check network adapter type
                result = subprocess.run(
                    ["netsh", "interface", "show", "interface"],
                    capture_output=True, text=True, timeout=2
                )
                if "Wi-Fi" in result.stdout and "Connected" in result.stdout:
                    mobility["is_mobile"] = False
                    mobility["connection_stability"] = 90
                elif "Cellular" in result.stdout or "Mobile" in result.stdout:
                    mobility["is_mobile"] = True
                    mobility["connection_stability"] = 60
        except:
            pass
        
        return mobility
    
    def calculate_protocol_scores(
        self,
        network: Dict[str, float],
        resources: Dict[str, float],
        model_size_mb: float,
        mobility: Dict[str, any]
    ) -> Dict[str, ProtocolScore]:
        """
        Calculate scores for all protocols based on current conditions.
        Returns dictionary of protocol scores.
        """
        scores = {}
        
        for protocol in ["mqtt", "amqp", "grpc", "quic", "dds"]:
            # Calculate component scores (0-100 scale)
            net_score = self._score_network_fit(protocol, network)
            res_score = self._score_resource_fit(protocol, resources)
            model_score = self._score_model_fit(protocol, model_size_mb)
            mob_score = self._score_mobility_fit(protocol, mobility)
            
            # Weighted total score
            total = (
                net_score * self.weights["network"] +
                res_score * self.weights["resources"] +
                model_score * self.weights["model"] +
                mob_score * self.weights["mobility"]
            )
            
            scores[protocol] = ProtocolScore(
                protocol=protocol,
                total_score=total,
                network_score=net_score,
                resource_score=res_score,
                model_score=model_score,
                mobility_score=mob_score,
                details=self._generate_score_details(
                    protocol, network, resources, model_size_mb, mobility
                )
            )
        
        return scores
    
    def select_best_protocol(
        self,
        model_size_mb: float = 10.0,
        verbose: bool = True
    ) -> Tuple[str, ProtocolScore]:
        """
        Main method: Select the best protocol for current conditions.
        
        Returns:
            (protocol_name, score_details)
        """
        # Gather current conditions
        network = self.measure_network_conditions()
        resources = self.measure_resource_availability()
        mobility = self.assess_mobility()
        
        # Calculate scores for all protocols
        scores = self.calculate_protocol_scores(network, resources, model_size_mb, mobility)
        
        # Select protocol with highest score
        best_protocol = max(scores.items(), key=lambda x: x[1].total_score)
        
        if verbose:
            print("\n" + "="*70)
            print("PROTOCOL SELECTION DECISION ENGINE")
            print("="*70)
            print(f"\n📊 Current Conditions:")
            print(f"  Network: Latency={network['latency']:.1f}ms, "
                  f"Loss={network['packet_loss']:.1f}%")
            print(f"  Resources: CPU={resources['cpu_usage']:.1f}%, "
                  f"Memory={resources['memory_percent']:.1f}%")
            print(f"  Model Size: {model_size_mb:.1f} MB")
            print(f"  Mobility: {'Mobile' if mobility['is_mobile'] else 'Stationary'}")
            
            print(f"\n🏆 Protocol Rankings:")
            for rank, (proto, score) in enumerate(
                sorted(scores.items(), key=lambda x: x[1].total_score, reverse=True), 1
            ):
                indicator = "✓ SELECTED" if proto == best_protocol[0] else ""
                print(f"  {rank}. {proto.upper():6s}: {score.total_score:5.1f} points {indicator}")
                print(f"     └─ Net:{score.network_score:4.1f} | "
                      f"Res:{score.resource_score:4.1f} | "
                      f"Model:{score.model_score:4.1f} | "
                      f"Mob:{score.mobility_score:4.1f}")
            
            print(f"\n✨ Selected Protocol: {best_protocol[0].upper()}")
            print(f"   Reason: {best_protocol[1].details['primary_reason']}")
            print("="*70 + "\n")
        
        return best_protocol[0], best_protocol[1]
    
    # ==================== Scoring Methods ====================
    
    def _score_network_fit(self, protocol: str, network: Dict[str, float]) -> float:
        """Score how well protocol fits network conditions (0-100)"""
        profile = self.protocol_profiles[protocol]
        
        # Determine network condition severity
        is_low_bandwidth = network.get('bandwidth', 100) < 5  # < 5 Mbps
        is_high_latency = network['latency'] > 100  # > 100ms
        has_packet_loss = network['packet_loss'] > 1  # > 1%
        
        score = 50  # Base score
        
        # Adjust based on conditions
        if is_high_latency:
            score = profile["high_latency"]
        
        if has_packet_loss:
            score = (score + profile["packet_loss"]) / 2
        
        if is_low_bandwidth:
            score = (score + profile["low_bandwidth"]) / 2
        
        return min(100, max(0, score))
    
    def _score_resource_fit(self, protocol: str, resources: Dict[str, float]) -> float:
        """Score how well protocol fits resource constraints (0-100)"""
        profile = self.protocol_profiles[protocol]
        
        is_low_cpu = resources['cpu_usage'] > 70  # High CPU usage
        is_low_memory = resources['memory_percent'] > 80  # Low memory
        is_low_battery = resources.get('battery_level', 100) < 30 and not resources.get('is_charging', True)
        
        score = 70  # Base score
        
        if is_low_cpu:
            score = (score + profile["low_cpu"]) / 2
        
        if is_low_memory:
            score = (score + profile["low_memory"]) / 2
        
        if is_low_battery:
            score = (score + profile["battery_efficient"]) / 2
        
        return min(100, max(0, score))
    
    def _score_model_fit(self, protocol: str, model_size_mb: float) -> float:
        """Score how well protocol handles model size (0-100)"""
        profile = self.protocol_profiles[protocol]
        
        is_large_model = model_size_mb > 50  # > 50 MB
        
        if is_large_model:
            return profile["large_model"]
        else:
            # Small models work with any protocol
            return 80
    
    def _score_mobility_fit(self, protocol: str, mobility: Dict[str, any]) -> float:
        """Score how well protocol handles mobility (0-100)"""
        profile = self.protocol_profiles[protocol]
        
        if mobility['is_mobile'] or mobility['connection_stability'] < 70:
            return profile["high_mobility"]
        else:
            # Stationary - all protocols work well
            return 80
    
    def _generate_score_details(
        self, protocol: str, network: Dict, resources: Dict,
        model_size_mb: float, mobility: Dict
    ) -> Dict[str, str]:
        """Generate human-readable explanation for score"""
        reasons = []
        
        # Network reasons
        if network['latency'] > 100:
            reasons.append("high latency tolerance")
        if network['packet_loss'] > 1:
            reasons.append("packet loss resilience")
        if network.get('bandwidth', 100) < 5:
            reasons.append("low bandwidth efficiency")
        
        # Resource reasons
        if resources['cpu_usage'] > 70:
            reasons.append("low CPU overhead")
        if resources['memory_percent'] > 80:
            reasons.append("low memory footprint")
        if resources.get('battery_level', 100) < 30:
            reasons.append("battery efficiency")
        
        # Model reasons
        if model_size_mb > 50:
            reasons.append("large model handling")
        
        # Mobility reasons
        if mobility['is_mobile']:
            reasons.append("mobility support")
        
        primary_reason = reasons[0] if reasons else "balanced performance"
        
        return {
            "primary_reason": primary_reason,
            "all_factors": ", ".join(reasons) if reasons else "optimal conditions"
        }
    
    # ==================== Measurement Helpers ====================
    
    def _measure_latency(self) -> float:
        """Measure latency to server (ping)"""
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["ping", "-n", "3", self.server_address],
                    capture_output=True, text=True, timeout=5
                )
                # Extract average from: "Average = XXms"
                match = re.search(r'Average = (\d+)ms', result.stdout)
                if match:
                    return float(match.group(1))
            else:  # Linux/Mac
                result = subprocess.run(
                    ["ping", "-c", "3", self.server_address],
                    capture_output=True, text=True, timeout=5
                )
                # Extract avg from: "min/avg/max"
                match = re.search(r'= [\d.]+/([\d.]+)/', result.stdout)
                if match:
                    return float(match.group(1))
        except:
            pass
        
        return 50.0  # Default if measurement fails
    
    def _estimate_bandwidth(self) -> float:
        """Estimate available bandwidth (simplified)"""
        # In production: Use iperf, speedtest, or network monitoring
        # For now, return a placeholder or check network interface stats
        
        try:
            net_io = psutil.net_io_counters()
            # Simple heuristic based on network interface
            # This is a placeholder - real implementation would measure actual throughput
            return 10.0  # Mbps (default assumption)
        except:
            return 10.0
    
    def _estimate_packet_loss(self) -> float:
        """Estimate packet loss percentage"""
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["ping", "-n", "10", self.server_address],
                    capture_output=True, text=True, timeout=10
                )
                # Extract loss from: "(X% loss)"
                match = re.search(r'\((\d+)% loss\)', result.stdout)
                if match:
                    return float(match.group(1))
        except:
            pass
        
        return 0.0  # Default: no loss
    
    def _get_battery_level(self) -> float:
        """Get battery level (0-100)"""
        try:
            battery = psutil.sensors_battery()
            if battery:
                return battery.percent
        except:
            pass
        
        return 100.0  # Default: assume plugged in
    
    def _is_charging(self) -> bool:
        """Check if device is charging"""
        try:
            battery = psutil.sensors_battery()
            if battery:
                return battery.power_plugged
        except:
            pass
        
        return True  # Default: assume plugged in


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Example 1: Simple selection
    selector = ProtocolSelector(server_address="localhost")
    protocol, score = selector.select_best_protocol(model_size_mb=25.0, verbose=True)
    
    print(f"\n🎯 Final Decision: Use {protocol.upper()} protocol")
    print(f"   Confidence Score: {score.total_score:.1f}/100")
    
    # Example 2: Simulate different scenarios
    print("\n" + "="*70)
    print("SCENARIO SIMULATIONS")
    print("="*70)
    
    scenarios = [
        ("Low bandwidth mobile", {"latency": 120, "packet_loss": 2, "bandwidth": 2}, 
         {"cpu_usage": 30, "battery_level": 40}, True),
        ("High performance server", {"latency": 10, "packet_loss": 0, "bandwidth": 100},
         {"cpu_usage": 20, "battery_level": 100}, False),
        ("Constrained IoT device", {"latency": 80, "packet_loss": 1, "bandwidth": 1},
         {"cpu_usage": 85, "battery_level": 20}, False),
    ]
    
    for name, network, resources, is_mobile in scenarios:
        scores = selector.calculate_protocol_scores(
            network={**{"jitter": 0}, **network},
            resources={**{"memory_available": 1000, "memory_percent": 50, "is_charging": False}, **resources},
            model_size_mb=25.0,
            mobility={"is_mobile": is_mobile, "velocity": 20 if is_mobile else 0, 
                     "connection_stability": 60 if is_mobile else 90, "handoffs_per_hour": 3 if is_mobile else 0}
        )
        
        best = max(scores.items(), key=lambda x: x[1].total_score)
        print(f"\n📱 Scenario: {name}")
        print(f"   Best Protocol: {best[0].upper()} (Score: {best[1].total_score:.1f})")
        print(f"   Reason: {best[1].details['primary_reason']}")
