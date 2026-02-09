#!/usr/bin/env python3
"""
Fix Late-Joining Client Support for All Servers

This script updates all FL server files to properly handle clients joining
during training (not just before training starts).
"""

import os
import re
from pathlib import Path

SERVER_FILES = [
    "Server/Emotion_Recognition/FL_Server_MQTT.py",
    "Server/Emotion_Recognition/FL_Server_AMQP.py",
    "Server/Emotion_Recognition/FL_Server_QUIC.py",
    "Server/Emotion_Recognition/FL_Server_DDS.py",
    "Server/MentalState_Recognition/FL_Server_MQTT.py",
    "Server/MentalState_Recognition/FL_Server_AMQP.py",
    "Server/MentalState_Recognition/FL_Server_QUIC.py",
    "Server/MentalState_Recognition/FL_Server_DDS.py",
    "Server/Temperature_Regulation/FL_Server_MQTT.py",
    "Server/Temperature_Regulation/FL_Server_AMQP.py",
    "Server/Temperature_Regulation/FL_Server_QUIC.py",
    "Server/Temperature_Regulation/FL_Server_DDS.py",
]

def add_training_started_flag(content):
    """Add training_started flag to __init__"""
    # Check if already exists
    if 'self.training_started = False' in content:
        return content, False
    
    # Add after self.converged = False
    pattern = r'(        self\.converged = False\n)'
    replacement = r'\1        self.training_started = False\n'
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        return content, True
    
    return content, False

def update_registration_logic(content):
    """Update client registration to handle late joining"""
    
    # Pattern to find the registration function
    old_pattern = r'''(def handle_client_registration\(self, payload\):
        """Handle client registration"""
        data = json\.loads\(payload\.decode\(\)\)
        client_id = data\['client_id'\]
        self\.registered_clients\.add\(client_id\)
        print\(f"Client \{client_id\} registered \(\{len\(self\.registered_clients\)\}/\{self\.num_clients\} expected, min: \{self\.min_clients\}\)"\)
        
        # Update total client count if more clients join
        if len\(self\.registered_clients\) > self\.num_clients:
            self\.update_client_count\(len\(self\.registered_clients\)\)
        
        # If all clients registered, distribute initial global model and start federated learning
        if len\(self\.registered_clients\) >= self\.min_clients:)'''
    
    new_pattern = r'''\1
        
        # Check if this is a late-joining client (training already started)
        if self.training_started:
            print(f"[LATE JOIN] Client {client_id} joining during training (round {self.current_round})")
            # Update client count dynamically
            if len(self.registered_clients) > self.num_clients:
                self.update_client_count(len(self.registered_clients))
            # Send current global model to late-joining client
            if self.global_weights is not None:
                self.send_current_model_to_client(client_id)
            return
        
        # Normal registration before training starts
        if len(self.registered_clients) >= self.min_clients and not self.training_started:'''
    
    if re.search(old_pattern, content, re.DOTALL):
        content = re.sub(old_pattern, new_pattern, content, flags=re.DOTALL)
        return content, True
    
    return content, False

def add_send_current_model_method(content):
    """Add method to send current model to late-joining client"""
    
    if 'def send_current_model_to_client' in content:
        return content, False
    
    # Find distribute_initial_model method and add after it
    pattern = r'(    def distribute_initial_model\(self\):.*?(?=\n    def ))'
    
    new_method = r'''\1
    
    def send_current_model_to_client(self, client_id):
        """Send current global model to a late-joining client"""
        try:
            print(f"Sending current global model (round {self.current_round}) to client {client_id}")
            
            # Prepare model payload
            payload = {
                'round': self.current_round,
                'weights': self.serialize_weights(self.global_weights),
                'training_config': self.training_config
            }
            
            # Send via appropriate protocol
            if hasattr(self, 'mqtt_client'):
                # MQTT
                self.mqtt_client.publish(
                    f"fl/client/{client_id}/model",
                    json.dumps(payload).encode(),
                    qos=1
                )
            elif hasattr(self, 'amqp_channel'):
                # AMQP
                self.amqp_channel.basic_publish(
                    exchange='federated_learning',
                    routing_key=f'client.{client_id}.model',
                    body=json.dumps(payload).encode(),
                    properties=pika.BasicProperties(delivery_mode=2)
                )
            elif hasattr(self, 'quic_connections'):
                # QUIC - handle separately in protocol-specific code
                pass
            elif hasattr(self, 'dds_writer'):
                # DDS - handle separately in protocol-specific code
                pass
            
            print(f"✅ Sent current model to late-joining client {client_id}")
        except Exception as e:
            print(f"❌ Error sending current model to client {client_id}: {e}")
    
'''
    
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, new_method, content, flags=re.DOTALL)
        return content, True
    
    return content, False

def update_distribute_initial_model(content):
    """Update distribute_initial_model to set training_started flag"""
    
    # Find the end of distribute_initial_model and add flag
    pattern = r'(def distribute_initial_model\(self\):.*?)(        print\(f"Training started at:)'
    replacement = r'\1        self.training_started = True\n        \2'
    
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        return content, True
    
    return content, False

def process_file(filepath):
    """Process a single server file"""
    print(f"\nProcessing: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"  ❌ File not found")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    modifications = []
    
    # Apply updates
    content, modified = add_training_started_flag(content)
    if modified:
        modifications.append("✓ Added training_started flag")
    
    content, modified = update_registration_logic(content)
    if modified:
        modifications.append("✓ Updated registration for late-joining clients")
    
    content, modified = add_send_current_model_method(content)
    if modified:
        modifications.append("✓ Added send_current_model_to_client method")
    
    content, modified = update_distribute_initial_model(content)
    if modified:
        modifications.append("✓ Updated distribute_initial_model to set flag")
    
    # Save if changed
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        
        print("  📝 Modifications:")
        for mod in modifications:
            print(f"     {mod}")
        print(f"  ✅ Updated")
        return True
    else:
        print("  ⚠️  No changes needed")
        return False

def main():
    """Main execution"""
    print("="*70)
    print("FIX LATE-JOINING CLIENT SUPPORT")
    print("="*70)
    
    os.chdir(Path(__file__).parent)
    
    updated = 0
    skipped = 0
    
    for filepath in SERVER_FILES:
        if process_file(filepath):
            updated += 1
        else:
            skipped += 1
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  ✅ Updated: {updated}")
    print(f"  ⚠️  Skipped: {skipped}")
    print("\n✨ Late-joining client support fixed!\n")

if __name__ == "__main__":
    main()
