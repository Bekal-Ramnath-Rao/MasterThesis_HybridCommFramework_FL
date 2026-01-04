#!/usr/bin/env python3
"""
Simple GUI/Interactive tool to add NET_ADMIN capability to docker-compose files
"""

import re
from pathlib import Path

def add_net_admin_to_compose(file_path):
    """Add cap_add: NET_ADMIN to all fl- services in a docker-compose file"""
    
    print(f"\nProcessing: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    modified = False
    i = 0
    
    while i < len(lines):
        line = lines[i]
        new_lines.append(line)
        
        # Check if this line contains "container_name: fl-"
        if re.match(r'\s+container_name:\s+fl-', line):
            # Check if next line already has cap_add
            if i + 1 < len(lines) and 'cap_add:' in lines[i + 1]:
                # Already has cap_add, check if NET_ADMIN is present
                if i + 2 < len(lines) and 'NET_ADMIN' not in lines[i + 2]:
                    # cap_add exists but no NET_ADMIN, add it
                    indent = len(lines[i + 2]) - len(lines[i + 2].lstrip())
                    new_lines.append(' ' * indent + '- NET_ADMIN\n')
                    modified = True
                    print(f"  ✓ Added NET_ADMIN to existing cap_add for: {line.strip()}")
            else:
                # No cap_add, add it with NET_ADMIN
                # Determine indentation (same as container_name line)
                indent = len(line) - len(line.lstrip())
                new_lines.append(' ' * indent + 'cap_add:\n')
                new_lines.append(' ' * (indent + 2) + '- NET_ADMIN\n')
                modified = True
                print(f"  ✓ Added cap_add with NET_ADMIN for: {line.strip()}")
        
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
    print("Docker Compose NET_ADMIN Capability Updater")
    print("="*70)
    print("\nThis tool adds 'cap_add: [NET_ADMIN]' to all FL services")
    print("in your docker-compose files for network simulation support.\n")
    
    # Find all docker-compose files
    compose_files = [
        'docker-compose-emotion.yml',
        'docker-compose-mentalstate.yml',
        'docker-compose-temperature.yml'
    ]
    
    existing_files = [f for f in compose_files if Path(f).exists()]
    
    if not existing_files:
        print("❌ No docker-compose files found in current directory!")
        print("Make sure you're running this from the project root.")
        return
    
    print(f"Found {len(existing_files)} docker-compose file(s):\n")
    for f in existing_files:
        print(f"  - {f}")
    
    print("\n" + "-"*70)
    response = input("\nProceed with updating these files? (y/n): ").strip().lower()
    
    if response != 'y':
        print("\n❌ Cancelled by user.")
        return
    
    print("\n" + "="*70)
    print("Processing files...")
    print("="*70)
    
    updated_count = 0
    for compose_file in existing_files:
        if add_net_admin_to_compose(compose_file):
            updated_count += 1
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Updated {updated_count} out of {len(existing_files)} file(s)")
    
    if updated_count > 0:
        print("\n✅ Success! Your docker-compose files are ready for network simulation.")
        print("\nNext steps:")
        print("1. Rebuild Docker images:")
        print("   docker-compose -f docker-compose-emotion.yml build")
        print("2. Test network simulation:")
        print("   python network_simulator.py --list")
        print("3. Run experiments:")
        print("   python run_network_experiments.py --help")
    else:
        print("\nℹ️  All files already have NET_ADMIN capability configured.")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
