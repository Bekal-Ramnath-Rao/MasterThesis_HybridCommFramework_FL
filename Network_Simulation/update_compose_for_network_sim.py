#!/usr/bin/env python3
"""
Update docker-compose files to add NET_ADMIN capability for network simulation
"""

import yaml
import sys
from pathlib import Path

def add_net_admin_capability(compose_file_path):
    """Add cap_add: NET_ADMIN to all FL server and client services"""
    
    print(f"\nProcessing: {compose_file_path}")
    
    with open(compose_file_path, 'r') as f:
        content = f.read()
    
    # Parse YAML
    try:
        compose_data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return False
    
    if 'services' not in compose_data:
        print("No services found in docker-compose file")
        return False
    
    modified = False
    
    # Add cap_add to all FL services (servers and clients)
    for service_name, service_config in compose_data['services'].items():
        # Skip brokers (mqtt-broker, rabbitmq)
        if 'broker' in service_name.lower() or 'rabbitmq' in service_name.lower():
            continue
        
        # Add cap_add if this is an FL service
        if service_name.startswith('fl-'):
            if 'cap_add' not in service_config:
                service_config['cap_add'] = ['NET_ADMIN']
                modified = True
                print(f"  ✓ Added NET_ADMIN to {service_name}")
            elif 'NET_ADMIN' not in service_config['cap_add']:
                service_config['cap_add'].append('NET_ADMIN')
                modified = True
                print(f"  ✓ Added NET_ADMIN to {service_name}")
            else:
                print(f"  - {service_name} already has NET_ADMIN")
    
    if modified:
        # Write back to file
        with open(compose_file_path, 'w') as f:
            yaml.dump(compose_data, f, default_flow_style=False, sort_keys=False, indent=2)
        print(f"  ✓ Updated {compose_file_path}")
        return True
    else:
        print(f"  - No changes needed for {compose_file_path}")
        return False

def main():
    # Find all docker-compose files
    compose_files = [
        'docker-compose-emotion.yml',
        'docker-compose-mentalstate.yml',
        'docker-compose-temperature.yml'
    ]
    
    print("="*70)
    print("Updating Docker Compose files for Network Simulation")
    print("="*70)
    
    updated_count = 0
    for compose_file in compose_files:
        path = Path(compose_file)
        if path.exists():
            if add_net_admin_capability(path):
                updated_count += 1
        else:
            print(f"\n[WARNING] File not found: {compose_file}")
    
    print("\n" + "="*70)
    print(f"Updated {updated_count} docker-compose file(s)")
    print("="*70)
    print("\nNext steps:")
    print("1. Rebuild Docker images: docker-compose build")
    print("2. Start containers: docker-compose up")
    print("3. Apply network conditions: python network_simulator.py --list")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
