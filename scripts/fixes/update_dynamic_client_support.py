#!/usr/bin/env python3
"""
Script to update FL Server for dynamic client support
Adds ability to handle clients joining mid-experiment and adaptive convergence
"""

import os
import sys

# Server files to update
SERVER_FILES = [
    "Server/Emotion_Recognition/FL_Server_Unified.py",
    "Server/MentalState_Recognition/FL_Server_Unified.py",
    "Server/Temperature_Regulation/FL_Server_Unified.py"
]

# Code modifications to add dynamic client support
DYNAMIC_CLIENT_ADDITIONS = """

    # =========================================================================
    # DYNAMIC CLIENT MANAGEMENT
    # =========================================================================
    
    def update_client_count(self, new_count):
        \"\"\"Update expected client count dynamically (thread-safe)\"\"\"
        with self.lock:
            old_count = self.num_clients
            self.num_clients = max(2, new_count)  # Minimum 2 clients
            print(f"[Server] Client count updated: {old_count} -> {self.num_clients}")
            
            # Reset convergence patience when client count changes
            if old_count != self.num_clients:
                self.rounds_without_improvement = 0
                print("[Server] Convergence patience reset due to client count change")
    
    def get_active_clients(self):
        \"\"\"Get list of currently active/registered clients\"\"\"
        with self.lock:
            return list(self.registered_clients.keys())
    
    def handle_late_joining_client(self, client_id, protocol):
        \"\"\"Handle a client that joins after training has started\"\"\"
        with self.lock:
            if client_id in self.registered_clients:
                # Client re-registering (e.g., after disconnect)
                print(f"[{protocol.upper()}] Client {client_id} re-registered")
                self.registered_clients[client_id] = protocol
            else:
                # New client joining mid-experiment
                print(f"[{protocol.upper()}] Late-joining client {client_id} registered")
                self.registered_clients[client_id] = protocol
                
                # Update expected client count
                new_count = len(self.registered_clients)
                if new_count > self.num_clients:
                    self.update_client_count(new_count)
                
                # Send current global model to new client immediately
                self.send_model_to_client(client_id, protocol)
                
                # If training has started, signal new client to start training
                if self.current_round > 0:
                    self.signal_start_training_single_client(client_id, protocol)
    
    def send_model_to_client(self, client_id, protocol):
        \"\"\"Send current global model to a specific client\"\"\"
        message = {
            'round': self.current_round,
            'weights': self.serialize_weights(self.global_weights),
            'model_config': self.model_config,
            'training_config': self.training_config
        }
        
        try:
            if protocol == 'mqtt':
                self.send_via_mqtt(client_id, f"fl/client/{client_id}/model", message)
            elif protocol == 'amqp':
                self.send_via_amqp(client_id, 'model', message)
            elif protocol == 'grpc':
                self.grpc_model_ready[client_id] = self.current_round
            elif protocol == 'quic':
                message['type'] = 'model'
                message['client_id'] = client_id
                self.send_quic_message(client_id, message)
            elif protocol == 'dds':
                self.send_via_dds(client_id, 'model', message)
            
            print(f"[{protocol.upper()}] Sent current model to late-joining client {client_id}")
        except Exception as e:
            print(f"[{protocol.upper()}] Error sending model to client {client_id}: {e}")
    
    def signal_start_training_single_client(self, client_id, protocol):
        \"\"\"Signal a single client to start training\"\"\"
        message = {
            'round': self.current_round,
            'action': 'train'
        }
        
        try:
            if protocol == 'mqtt':
                self.mqtt_client.publish(f"fl/client/{client_id}/train", json.dumps(message), qos=1)
            elif protocol == 'amqp':
                self.send_via_amqp(client_id, 'train', message)
            elif protocol == 'grpc':
                self.grpc_should_train[client_id] = True
            elif protocol == 'quic':
                message['type'] = 'train'
                message['client_id'] = client_id
                self.send_quic_message(client_id, message)
            elif protocol == 'dds':
                self.send_via_dds(client_id, 'train', message)
            
            print(f"[{protocol.upper()}] Signaled client {client_id} to start training")
        except Exception as e:
            print(f"[{protocol.upper()}] Error signaling client {client_id}: {e}")
    
    def adaptive_wait_for_clients(self, timeout=300):
        \"\"\"
        Adaptively wait for clients with dynamic timeout
        Returns True if enough clients joined, False if timeout
        \"\"\"
        import time
        start_time = time.time()
        min_clients = max(2, int(self.num_clients * 0.5))  # At least 50% of expected clients
        
        while time.time() - start_time < timeout:
            with self.lock:
                current_count = len(self.registered_clients)
                
                # If we have all expected clients, return immediately
                if current_count >= self.num_clients:
                    return True
                
                # If we have minimum required clients and waited > 30s, proceed
                if current_count >= min_clients and time.time() - start_time > 30:
                    print(f"[Server] Proceeding with {current_count}/{self.num_clients} clients")
                    self.update_client_count(current_count)
                    return True
            
            time.sleep(1)
        
        # Timeout reached
        with self.lock:
            current_count = len(self.registered_clients)
            if current_count >= min_clients:
                print(f"[Server] Timeout reached, proceeding with {current_count} clients")
                self.update_client_count(current_count)
                return True
            else:
                print(f"[Server] Timeout reached with only {current_count} clients (min: {min_clients})")
                return False
"""

def update_server_file(filepath):
    """Update a server file with dynamic client support"""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found, skipping...")
        return False
    
    print(f"Updating {filepath}...")
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check if already updated
        if "handle_late_joining_client" in content:
            print(f"  ✓ {filepath} already has dynamic client support")
            return True
        
        # Find the handle_client_registration method and update it
        if "def handle_client_registration(self, client_id, protocol):" in content:
            # Replace the registration logic to use late-joining handler
            old_registration = """    def handle_client_registration(self, client_id, protocol):
        \"\"\"Handle client registration (thread-safe)\"\"\"
        with self.lock:
            self.registered_clients[client_id] = protocol
            print(f"[{protocol.upper()}] Client {client_id} registered "
                  f"({len(self.registered_clients)}/{self.num_clients})")
            
            if len(self.registered_clients) == self.num_clients:
                print(f"\\n[Server] All {self.num_clients} clients registered!")
                print("[Server] Distributing initial global model...\\n")
                time.sleep(2)
                self.distribute_initial_model()
                self.start_time = time.time()
                print(f"[Server] Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\\n")"""
            
            new_registration = """    def handle_client_registration(self, client_id, protocol):
        \"\"\"Handle client registration (thread-safe, supports late joining)\"\"\"
        if self.current_round > 0:
            # Training already started - late joining client
            self.handle_late_joining_client(client_id, protocol)
        else:
            # Initial registration phase
            with self.lock:
                self.registered_clients[client_id] = protocol
                print(f"[{protocol.upper()}] Client {client_id} registered "
                      f"({len(self.registered_clients)}/{self.num_clients})")
                
                if len(self.registered_clients) >= self.num_clients:
                    print(f"\\n[Server] All {self.num_clients} clients registered!")
                    print("[Server] Distributing initial global model...\\n")
                    time.sleep(2)
                    self.distribute_initial_model()
                    self.start_time = time.time()
                    print(f"[Server] Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\\n")"""
            
            content = content.replace(old_registration, new_registration)
            
            # Add the new dynamic client methods after the handle_client_registration method
            # Find a good insertion point (after handle_client_metrics)
            insertion_point = content.find("    def handle_client_update(self, data, protocol):")
            if insertion_point > 0:
                content = content[:insertion_point] + DYNAMIC_CLIENT_ADDITIONS + "\n" + content[insertion_point:]
            
            # Update aggregate_models to use actual client count
            old_aggregate = """    def aggregate_models(self):
        \"\"\"Aggregate client model updates using FedAvg\"\"\"
        print(f"\\n{'='*70}")
        print(f"ROUND {self.current_round}/{self.num_rounds} - AGGREGATING MODELS")
        print(f"{'='*70}")"""
            
            new_aggregate = """    def aggregate_models(self):
        \"\"\"Aggregate client model updates using FedAvg (dynamic client support)\"\"\"
        with self.lock:
            num_active_clients = len(self.registered_clients)
        
        print(f"\\n{'='*70}")
        print(f"ROUND {self.current_round}/{self.num_rounds} - AGGREGATING MODELS")
        print(f"Active Clients: {num_active_clients}")
        print(f"{'='*70}")"""
            
            content = content.replace(old_aggregate, new_aggregate)
            
            # Write updated content
            with open(filepath, 'w') as f:
                f.write(content)
            
            print(f"  ✓ Successfully updated {filepath}")
            return True
        else:
            print(f"  ✗ Could not find registration method in {filepath}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error updating {filepath}: {e}")
        return False

def main():
    print("=" * 70)
    print("FL Server Dynamic Client Support Updater")
    print("=" * 70)
    print()
    
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir
    
    success_count = 0
    total_count = 0
    
    for server_file in SERVER_FILES:
        total_count += 1
        filepath = os.path.join(project_root, server_file)
        
        if update_server_file(filepath):
            success_count += 1
        print()
    
    print("=" * 70)
    print(f"Update Summary: {success_count}/{total_count} files updated successfully")
    print("=" * 70)
    print()
    
    if success_count > 0:
        print("✓ Server files updated with dynamic client support")
        print()
        print("New features:")
        print("  • Clients can join mid-experiment")
        print("  • Server adapts to variable client count (min 2)")
        print("  • Late-joining clients receive current model")
        print("  • Convergence resets when client count changes")
        print()
        print("Note: You may need to rebuild Docker images for changes to take effect")
        print("Run: docker-compose -f Docker/docker-compose-unified-<usecase>.yml build")
    
    return 0 if success_count == total_count else 1

if __name__ == "__main__":
    sys.exit(main())
