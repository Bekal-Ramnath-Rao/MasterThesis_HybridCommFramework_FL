"""
Dynamic FL Client with Protocol Selection
Automatically selects the best communication protocol based on current conditions.
"""

import os
import sys
import time
from protocol_selector import ProtocolSelector, ProtocolScore


class DynamicFLClient:
    """
    FL Client that dynamically selects communication protocol.
    
    Features:
    - Periodic protocol re-evaluation (every N rounds)
    - Automatic protocol switching
    - Performance monitoring
    """
    
    def __init__(
        self,
        client_id: int,
        server_address: str = "localhost",
        model_size_mb: float = 25.0,
        reevaluate_every: int = 5  # Re-evaluate protocol every N rounds
    ):
        self.client_id = client_id
        self.server_address = server_address
        self.model_size_mb = model_size_mb
        self.reevaluate_every = reevaluate_every
        
        # Protocol selector
        self.selector = ProtocolSelector(server_address=server_address)
        
        # Current protocol and client instance
        self.current_protocol = None
        self.client_instance = None
        
        # Performance tracking
        self.round_count = 0
        self.protocol_switches = []
        
    def initialize(self):
        """Initialize client with best protocol"""
        print(f"\n{'='*70}")
        print(f"DYNAMIC FL CLIENT {self.client_id} - INITIALIZATION")
        print(f"{'='*70}\n")
        
        # Select initial protocol
        protocol, score = self.selector.select_best_protocol(
            model_size_mb=self.model_size_mb,
            verbose=True
        )
        
        self.switch_protocol(protocol, reason="initial_selection")
        
    def switch_protocol(self, new_protocol: str, reason: str = "optimization"):
        """Switch to a different protocol"""
        if new_protocol == self.current_protocol:
            print(f"✓ Continuing with {new_protocol.upper()} protocol")
            return
        
        print(f"\n🔄 Protocol Switch: {self.current_protocol or 'None'} → {new_protocol.upper()}")
        print(f"   Reason: {reason}")
        
        # Track switch
        self.protocol_switches.append({
            "round": self.round_count,
            "from": self.current_protocol,
            "to": new_protocol,
            "reason": reason
        })
        
        # Clean up old client
        if self.client_instance:
            try:
                self.client_instance.disconnect()
            except:
                pass
        
        # Initialize new protocol client
        self.current_protocol = new_protocol
        self.client_instance = self._create_client_instance(new_protocol)
        
        print(f"✓ {new_protocol.upper()} client initialized\n")
    
    def _create_client_instance(self, protocol: str):
        """Create FL client instance for specific protocol"""
        # Import appropriate client based on protocol
        use_case = os.getenv("USE_CASE", "emotion")
        
        if protocol == "mqtt":
            if use_case == "emotion":
                from Emotion_Recognition.FL_Client_MQTT import FederatedLearningClient
            elif use_case == "mentalstate":
                from MentalState_Recognition.FL_Client_MQTT import FederatedLearningClient
            else:  # temperature
                from Temperature_Regulation.FL_Client_MQTT import FederatedLearningClient
                
        elif protocol == "amqp":
            if use_case == "emotion":
                from Emotion_Recognition.FL_Client_AMQP import FederatedLearningClient
            elif use_case == "mentalstate":
                from MentalState_Recognition.FL_Client_AMQP import FederatedLearningClient
            else:
                from Temperature_Regulation.FL_Client_AMQP import FederatedLearningClient
                
        elif protocol == "grpc":
            if use_case == "emotion":
                from Emotion_Recognition.FL_Client_gRPC import FederatedLearningClient
            elif use_case == "mentalstate":
                from MentalState_Recognition.FL_Client_gRPC import FederatedLearningClient
            else:
                from Temperature_Regulation.FL_Client_gRPC import FederatedLearningClient
                
        elif protocol == "quic":
            if use_case == "emotion":
                from Emotion_Recognition.FL_Client_QUIC import FederatedLearningClient
            elif use_case == "mentalstate":
                from MentalState_Recognition.FL_Client_QUIC import FederatedLearningClient
            else:
                from Temperature_Regulation.FL_Client_QUIC import FederatedLearningClient
                
        else:  # dds
            if use_case == "emotion":
                from Emotion_Recognition.FL_Client_DDS import FederatedLearningClient
            elif use_case == "mentalstate":
                from MentalState_Recognition.FL_Client_DDS import FederatedLearningClient
            else:
                from Temperature_Regulation.FL_Client_DDS import FederatedLearningClient
        
        # Create and return client instance
        return FederatedLearningClient(
            client_id=self.client_id,
            num_clients=int(os.getenv("NUM_CLIENTS", 2))
        )
    
    def should_reevaluate_protocol(self) -> bool:
        """Determine if it's time to re-evaluate protocol"""
        return self.round_count % self.reevaluate_every == 0
    
    def reevaluate_protocol(self):
        """Re-evaluate and potentially switch protocol"""
        print(f"\n{'='*70}")
        print(f"PROTOCOL RE-EVALUATION (Round {self.round_count})")
        print(f"{'='*70}\n")
        
        # Get new recommendation
        protocol, score = self.selector.select_best_protocol(
            model_size_mb=self.model_size_mb,
            verbose=True
        )
        
        # Check if switch is needed
        if protocol != self.current_protocol:
            # Switch only if new protocol has significantly better score
            # (avoid frequent switching)
            if score.total_score > 10:  # At least 10 points better
                self.switch_protocol(protocol, reason="optimization")
            else:
                print(f"✓ Staying with {self.current_protocol.upper()} "
                      f"(new protocol not significantly better)")
        
    def run_training_round(self, round_num: int):
        """Execute one training round"""
        self.round_count = round_num
        
        # Periodic protocol re-evaluation
        if self.should_reevaluate_protocol() and round_num > 0:
            self.reevaluate_protocol()
        
        # Execute training with current protocol
        print(f"\n[Round {round_num}] Training with {self.current_protocol.upper()}...")
        # Actual training logic here...
        # self.client_instance.train()
        
    def get_statistics(self):
        """Get statistics about protocol usage"""
        stats = {
            "total_rounds": self.round_count,
            "protocol_switches": len(self.protocol_switches),
            "current_protocol": self.current_protocol,
            "switches": self.protocol_switches
        }
        
        print(f"\n{'='*70}")
        print(f"CLIENT {self.client_id} - PROTOCOL USAGE STATISTICS")
        print(f"{'='*70}")
        print(f"Total Rounds: {stats['total_rounds']}")
        print(f"Protocol Switches: {stats['protocol_switches']}")
        print(f"Current Protocol: {stats['current_protocol'].upper()}")
        
        if stats['switches']:
            print(f"\nSwitch History:")
            for switch in stats['switches']:
                print(f"  Round {switch['round']}: "
                      f"{switch['from'] or 'None'} → {switch['to']} "
                      f"({switch['reason']})")
        
        print(f"{'='*70}\n")
        
        return stats


# ==================== Usage Example ====================

if __name__ == "__main__":
    # Example: Dynamic client that adapts to conditions
    
    print("Dynamic Federated Learning Client Demo")
    print("Simulating protocol selection across different scenarios\n")
    
    # Create client
    client = DynamicFLClient(
        client_id=1,
        server_address="localhost",
        model_size_mb=25.0,
        reevaluate_every=3  # Re-evaluate every 3 rounds
    )
    
    # Initialize
    client.initialize()
    
    # Simulate training rounds
    print(f"\n{'='*70}")
    print("SIMULATING TRAINING ROUNDS")
    print(f"{'='*70}\n")
    
    for round_num in range(1, 10):
        client.run_training_round(round_num)
        time.sleep(1)  # Simulate training time
    
    # Show statistics
    client.get_statistics()
