#!/usr/bin/env python3
"""
Update All FL Servers for Dynamic Client Support

This script updates all FL server files (MQTT, AMQP, gRPC, QUIC, DDS, Unified)
across all use cases (Emotion, Mental State, Temperature) to support:
1. Dynamic client registration (more than 2 clients)
2. Minimum client threshold for starting training
3. All registered clients included in aggregation and evaluation
"""

import os
import re
import sys
from pathlib import Path

# Server file patterns
SERVER_FILES = [
    # Emotion Recognition
    "Server/Emotion_Recognition/FL_Server_MQTT.py",
    "Server/Emotion_Recognition/FL_Server_AMQP.py",
    "Server/Emotion_Recognition/FL_Server_gRPC.py",
    "Server/Emotion_Recognition/FL_Server_QUIC.py",
    "Server/Emotion_Recognition/FL_Server_DDS.py",
    "Server/Emotion_Recognition/FL_Server_Unified.py",
    # Mental State Recognition
    "Server/MentalState_Recognition/FL_Server_MQTT.py",
    "Server/MentalState_Recognition/FL_Server_AMQP.py",
    "Server/MentalState_Recognition/FL_Server_gRPC.py",
    "Server/MentalState_Recognition/FL_Server_QUIC.py",
    "Server/MentalState_Recognition/FL_Server_DDS.py",
    "Server/MentalState_Recognition/FL_Server_Unified.py",
    # Temperature Regulation
    "Server/Temperature_Regulation/FL_Server_MQTT.py",
    "Server/Temperature_Regulation/FL_Server_AMQP.py",
    "Server/Temperature_Regulation/FL_Server_gRPC.py",
    "Server/Temperature_Regulation/FL_Server_QUIC.py",
    "Server/Temperature_Regulation/FL_Server_DDS.py",
    "Server/Temperature_Regulation/FL_Server_Unified.py",
]

def backup_file(filepath):
    """Create backup of original file"""
    backup_path = f"{filepath}.bak_dynamic_clients"
    if not os.path.exists(backup_path):
        with open(filepath, 'r') as f:
            content = f.read()
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"  ✅ Backup created: {backup_path}")
    else:
        print(f"  ⚠️ Backup already exists: {backup_path}")

def update_num_clients_config(content):
    """Update NUM_CLIENTS configuration to use MIN_CLIENTS"""
    # Replace hardcoded NUM_CLIENTS with MIN_CLIENTS
    pattern = r'NUM_CLIENTS = int\(os\.getenv\("NUM_CLIENTS", "[0-9]+"\)\)'
    replacement = '''# Dynamic client configuration
MIN_CLIENTS = int(os.getenv("MIN_CLIENTS", "2"))  # Minimum clients to start training
MAX_CLIENTS = int(os.getenv("MAX_CLIENTS", "100"))  # Maximum clients allowed'''
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        return content, True
    return content, False

def update_init_method(content):
    """Update __init__ method to use min_clients"""
    # Pattern 1: def __init__(self, num_clients, num_rounds):
    pattern1 = r'def __init__\(self, num_clients, num_rounds\):'
    replacement1 = 'def __init__(self, min_clients, num_rounds, max_clients=100):'
    
    # Pattern 2: self.num_clients = num_clients
    pattern2 = r'self\.num_clients = num_clients'
    replacement2 = '''self.min_clients = min_clients
        self.max_clients = max_clients
        self.num_clients = min_clients  # Start with minimum, will update as clients join'''
    
    modified = False
    if re.search(pattern1, content):
        content = re.sub(pattern1, replacement1, content)
        modified = True
    
    if re.search(pattern2, content):
        content = re.sub(pattern2, replacement2, content)
        modified = True
    
    return content, modified

def add_dynamic_client_methods(content):
    """Add methods for handling dynamic client registration"""
    # Check if methods already exist
    if 'def update_client_count' in content:
        return content, False
    
    # Find a good insertion point (after __init__ method)
    init_pattern = r'(def __init__.*?(?=\n    def |\nclass |\Z))'
    
    dynamic_methods = '''
    
    def update_client_count(self, new_count):
        """Update expected client count when new clients join"""
        if new_count > self.num_clients and new_count <= self.max_clients:
            old_count = self.num_clients
            self.num_clients = new_count
            print(f"[DYNAMIC] Updated client count: {old_count} -> {new_count}")
            return True
        return False
    
    def handle_late_joining_client(self, client_id):
        """Handle a client joining after training has started"""
        if client_id not in self.registered_clients:
            self.registered_clients.add(client_id)
            print(f"[LATE JOIN] Client {client_id} joined after training started")
            
            # Update client count if needed
            if len(self.registered_clients) > self.num_clients:
                self.update_client_count(len(self.registered_clients))
            
            # Send current global model to late-joining client
            if self.global_weights is not None:
                self.send_global_model_to_client(client_id)
    
    def get_active_clients(self):
        """Get list of currently active clients"""
        return list(self.registered_clients)
    
    def adaptive_wait_for_clients(self, client_dict, timeout=300):
        """
        Adaptive waiting for client responses
        - Waits for minimum clients first
        - Then waits additional time for late-joining clients
        - Returns when all registered clients respond or timeout
        """
        import time
        start_time = time.time()
        min_received = len(client_dict) >= self.min_clients
        all_registered_received = len(client_dict) >= len(self.registered_clients)
        
        while not all_registered_received and (time.time() - start_time) < timeout:
            time.sleep(0.5)
            min_received = len(client_dict) >= self.min_clients
            all_registered_received = len(client_dict) >= len(self.registered_clients)
            
            # If we have minimum and haven't seen new clients for 10 seconds, proceed
            if min_received and (time.time() - start_time) > 10:
                break
        
        return len(client_dict) >= self.min_clients
'''
    
    # Find the end of __init__ method
    match = re.search(r'(        # Training timeout tracking.*?\n\n)', content, re.DOTALL)
    if match:
        insert_pos = match.end()
        content = content[:insert_pos] + dynamic_methods + content[insert_pos:]
        return content, True
    
    # Fallback: insert after quantization initialization
    match = re.search(r'(        else:\n            print\("Server: Quantization disabled"\)\n)', content)
    if match:
        insert_pos = match.end()
        content = content[:insert_pos] + dynamic_methods + content[insert_pos:]
        return content, True
    
    return content, False

def update_client_registration(content):
    """Update client registration to handle dynamic clients"""
    # Update registration check to use min_clients
    pattern1 = r'if len\(self\.registered_clients\) == self\.num_clients:'
    replacement1 = 'if len(self.registered_clients) >= self.min_clients:'
    
    # Update client count display
    pattern2 = r'print\(f"Client \{client_id\} registered \(\{len\(self\.registered_clients\)\}/\{self\.num_clients\}\)"\)'
    replacement2 = '''print(f"Client {client_id} registered ({len(self.registered_clients)}/{self.num_clients} expected, min: {self.min_clients})")
        
        # Update total client count if more clients join
        if len(self.registered_clients) > self.num_clients:
            self.update_client_count(len(self.registered_clients))'''
    
    modified = False
    if re.search(pattern1, content):
        content = re.sub(pattern1, replacement1, content)
        modified = True
    
    if re.search(pattern2, content):
        content = re.sub(pattern2, replacement2, content)
        modified = True
    
    return content, modified

def update_aggregation_logic(content):
    """Update aggregation to use all registered clients"""
    # Update aggregation waiting logic
    pattern = r'if len\(self\.client_updates\) == self\.num_clients:'
    replacement = '''# Wait for all registered clients (dynamic)
            if len(self.client_updates) >= len(self.registered_clients):'''
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        return content, True
    
    return content, False

def update_evaluation_logic(content):
    """Update evaluation to include all registered clients"""
    # Update evaluation waiting logic
    pattern = r'if len\(self\.client_metrics\) == self\.num_clients:'
    replacement = '''# Wait for all registered clients (dynamic)
            if len(self.client_metrics) >= len(self.registered_clients):'''
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        return content, True
    
    return content, False

def update_main_initialization(content):
    """Update main() to pass min_clients and max_clients"""
    # Update server initialization
    pattern = r'server = FederatedLearningServer\(NUM_CLIENTS, NUM_ROUNDS\)'
    replacement = 'server = FederatedLearningServer(MIN_CLIENTS, NUM_ROUNDS, MAX_CLIENTS)'
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        return content, True
    
    return content, False

def update_logging_statements(content):
    """Update logging to show min/max/current clients"""
    # Update client count logging in main()
    pattern = r'print\(f"Clients: \{NUM_CLIENTS\}"\)'
    replacement = 'print(f"Clients: {MIN_CLIENTS} (min) - {MAX_CLIENTS} (max)")'
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        return content, True
    
    return content, False

def process_file(filepath):
    """Process a single server file"""
    print(f"\n{'='*70}")
    print(f"Processing: {filepath}")
    print('='*70)
    
    if not os.path.exists(filepath):
        print(f"  ❌ File not found: {filepath}")
        return False
    
    # Backup original file
    backup_file(filepath)
    
    # Read file
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    modifications = []
    
    # Apply updates
    content, modified = update_num_clients_config(content)
    if modified:
        modifications.append("✓ Updated NUM_CLIENTS -> MIN_CLIENTS/MAX_CLIENTS config")
    
    content, modified = update_init_method(content)
    if modified:
        modifications.append("✓ Updated __init__ signature and attributes")
    
    content, modified = add_dynamic_client_methods(content)
    if modified:
        modifications.append("✓ Added dynamic client handling methods")
    
    content, modified = update_client_registration(content)
    if modified:
        modifications.append("✓ Updated client registration logic")
    
    content, modified = update_aggregation_logic(content)
    if modified:
        modifications.append("✓ Updated aggregation to use all registered clients")
    
    content, modified = update_evaluation_logic(content)
    if modified:
        modifications.append("✓ Updated evaluation to include all registered clients")
    
    content, modified = update_main_initialization(content)
    if modified:
        modifications.append("✓ Updated server initialization in main()")
    
    content, modified = update_logging_statements(content)
    if modified:
        modifications.append("✓ Updated logging statements")
    
    # Write updated content
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        
        print("\n  📝 Modifications applied:")
        for mod in modifications:
            print(f"     {mod}")
        print(f"\n  ✅ Successfully updated: {filepath}")
        return True
    else:
        print("  ⚠️ No changes needed (already updated or different structure)")
        return False

def main():
    """Main execution"""
    print("="*70)
    print("FL SERVER DYNAMIC CLIENT SUPPORT UPDATE")
    print("="*70)
    print("\nThis script will update all FL server files to support:")
    print("  • Dynamic client registration (more than 2 clients)")
    print("  • Minimum client threshold for starting training")
    print("  • All registered clients in aggregation and evaluation")
    print("\nBackups will be created as: <filename>.bak_dynamic_clients")
    
    response = input("\nProceed with updates? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Aborted.")
        return
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    updated_count = 0
    failed_count = 0
    skipped_count = 0
    
    for filepath in SERVER_FILES:
        try:
            if process_file(filepath):
                updated_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"  ❌ Error processing {filepath}: {e}")
            failed_count += 1
    
    # Summary
    print("\n" + "="*70)
    print("UPDATE SUMMARY")
    print("="*70)
    print(f"  ✅ Updated:  {updated_count} files")
    print(f"  ⚠️  Skipped:  {skipped_count} files")
    print(f"  ❌ Failed:   {failed_count} files")
    print("\n" + "="*70)
    
    if updated_count > 0:
        print("\n🎯 NEXT STEPS:")
        print("  1. Review the changes in the updated files")
        print("  2. Rebuild Docker images:")
        print("     cd Docker")
        print("     docker-compose -f docker-compose-<usecase>.gpu-isolated.yml build")
        print("     docker-compose -f docker-compose-unified-<usecase>.yml build")
        print("  3. Test with distributed clients from the GUI")
        print("\n  💡 To revert changes, restore from .bak_dynamic_clients files")
    
    print("\n✨ Update complete!\n")

if __name__ == "__main__":
    main()
