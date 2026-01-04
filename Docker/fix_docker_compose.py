#!/usr/bin/env python3
"""
Properly add cap_add and volumes to docker-compose files
"""

import yaml
from pathlib import Path

def update_docker_compose(input_file, output_file, dataset_path):
    """Update docker-compose file with cap_add and dataset volumes"""
    
    with open(input_file, 'r') as f:
        compose = yaml.safe_load(f)
    
    if 'services' not in compose:
        print(f"No services found in {input_file}")
        return False
    
    modified_count = 0
    
    for service_name, service_config in compose['services'].items():
        # Add cap_add to all FL services (not brokers)
        if service_name.startswith('fl-'):
            # Add NET_ADMIN capability
            if 'cap_add' not in service_config:
                service_config['cap_add'] = []
            if 'NET_ADMIN' not in service_config['cap_add']:
                service_config['cap_add'].append('NET_ADMIN')
                modified_count += 1
                print(f"  + Added NET_ADMIN to {service_name}")
            
            # Add dataset volume to clients
            if 'client' in service_name:
                if 'volumes' not in service_config:
                    service_config['volumes'] = []
                
                # Add dataset mount
                dataset_mount = f"{dataset_path}:/app/{dataset_path}"
                if dataset_mount not in service_config['volumes']:
                    service_config['volumes'].append(dataset_mount)
                    modified_count += 1
                    print(f"  + Added dataset volume to {service_name}")
                
                # Add certs for QUIC/gRPC clients
                if 'quic' in service_name or 'grpc' in service_name:
                    certs_mount = "./certs:/app/certs"
                    if certs_mount not in service_config['volumes']:
                        service_config['volumes'].append(certs_mount)
                        print(f"  + Added certs volume to {service_name}")
    
    # Write output
    with open(output_file, 'w') as f:
        yaml.dump(compose, f, default_flow_style=False, sort_keys=False, indent=2)
    
    print(f"\n✅ Updated {output_file} ({modified_count} modifications)")
    return True

def main():
    print("="*70)
    print("Docker Compose Updater - Clean Version")
    print("="*70 + "\n")
    
    # Update Emotion Recognition
    print("Updating Emotion Recognition docker-compose...")
    update_docker_compose(
        'docker-compose-emotion.yml',
        'docker-compose-emotion.yml',
        'Client/Emotion_Recognition/Dataset'
    )
    
    print("\n" + "="*70)
    print("✅ Complete! Now restart your containers:")
    print("   docker-compose -f docker-compose-emotion.yml down")
    print("   docker-compose -f docker-compose-emotion.yml up -d")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
