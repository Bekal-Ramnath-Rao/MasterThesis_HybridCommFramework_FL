#!/usr/bin/env python3
"""
Add dataset volume mounts to all client services in docker-compose files
"""

import re
from pathlib import Path

def add_volumes_to_clients(file_path, use_case):
    """Add volume mounts for datasets to all client services"""
    
    print(f"\nProcessing: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    modified = False
    i = 0
    
    # Map use case to dataset path
    dataset_paths = {
        "emotion": "./Client/Emotion_Recognition/Dataset:/app/Client/Emotion_Recognition/Dataset",
        "mental": "./Client/MentalState_Recognition/Dataset:/app/Client/MentalState_Recognition/Dataset",
        "temp": "./Client/Temperature_Regulation/Dataset:/app/Client/Temperature_Regulation/Dataset"
    }
    
    volume_mount = dataset_paths.get(use_case, dataset_paths["emotion"])
    
    while i < len(lines):
        line = lines[i]
        new_lines.append(line)
        
        # Check if this is a client service
        if re.match(r'\s+container_name:\s+fl-client-', line):
            # Look ahead to find where to insert volumes
            # Skip environment, command sections and find networks
            j = i + 1
            has_volumes = False
            networks_line_idx = None
            
            while j < len(lines):
                if 'volumes:' in lines[j]:
                    has_volumes = True
                    break
                if re.match(r'\s+networks:', lines[j]):
                    networks_line_idx = j
                    break
                if re.match(r'^\s{2}\w+:', lines[j]):  # Next service definition
                    break
                j += 1
            
            if not has_volumes and networks_line_idx:
                # Add volumes section before networks
                indent = len(lines[networks_line_idx]) - len(lines[networks_line_idx].lstrip())
                
                # Count remaining lines to add
                remaining_lines = lines[i+1:]
                
                # Add all lines up to networks
                for k in range(i+1, networks_line_idx):
                    new_lines.append(lines[k])
                
                # Insert volumes section
                new_lines.append(' ' * indent + 'volumes:\n')
                new_lines.append(' ' * (indent + 2) + f'- {volume_mount}\n')
                modified = True
                print(f"  ✓ Added volume mount for: {line.strip()}")
                
                # Add networks and rest
                i = networks_line_idx - 1  # Will be incremented at end of loop
            elif has_volumes:
                print(f"  - {line.strip()} already has volumes section")
        
        i += 1
    
    if modified:
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"  ✅ Successfully updated {file_path}")
        return True
    else:
        print(f"  ℹ️  No changes needed for {file_path}")
        return False


def main():
    print("="*70)
    print("Docker Compose Dataset Volume Mounts Updater")
    print("="*70)
    print("\nAdding dataset volume mounts to all client services...\n")
    
    # Map files to use cases
    file_mappings = [
        ('docker-compose-emotion.yml', 'emotion'),
        ('docker-compose-mentalstate.yml', 'mental'),
        ('docker-compose-temperature.yml', 'temp')
    ]
    
    existing_files = [(f, u) for f, u in file_mappings if Path(f).exists()]
    
    if not existing_files:
        print("❌ No docker-compose files found!")
        return
    
    print(f"Found {len(existing_files)} docker-compose file(s)\n")
    
    updated_count = 0
    for compose_file, use_case in existing_files:
        if add_volumes_to_clients(compose_file, use_case):
            updated_count += 1
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Updated {updated_count} out of {len(existing_files)} file(s)")
    
    if updated_count > 0:
        print("\n✅ Success! Client services now have dataset access.")
        print("\nNext step: Restart your containers:")
        print("   docker-compose -f docker-compose-emotion.yml down")
        print("   docker-compose -f docker-compose-emotion.yml up -d fl-server-mqtt-emotion fl-client-mqtt-emotion-1 fl-client-mqtt-emotion-2")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
