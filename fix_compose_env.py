#!/usr/bin/env python3
"""Fix docker-compose files by properly adding MIN_CLIENTS and MAX_CLIENTS"""

import re
from pathlib import Path

compose_files = [
    "Docker/docker-compose-unified-mentalstate.yml",
    "Docker/docker-compose-unified-temperature.yml",
    "Docker/docker-compose-emotion.gpu-isolated.yml",
    "Docker/docker-compose-mentalstate.gpu-isolated.yml",
    "Docker/docker-compose-temperature.gpu-isolated.yml",
]

for filepath in compose_files:
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Pattern: Find NUM_ROUNDS line and add MIN/MAX_CLIENTS after it
        # Only if not already present
        if 'MIN_CLIENTS' in content:
            print(f"⏭️  {filepath} - Already has MIN_CLIENTS")
            continue
        
        # Replace NUM_ROUNDS=... with NUM_ROUNDS + MIN/MAX_CLIENTS
        pattern = r'(      - NUM_ROUNDS=\d+)\n(      - )'
        replacement = r'\1\n      - MIN_CLIENTS=${MIN_CLIENTS:-2}\n      - MAX_CLIENTS=${MAX_CLIENTS:-100}\n\2'
        
        new_content = re.sub(pattern, replacement, content)
        
        if new_content != content:
            with open(filepath, 'w') as f:
                f.write(new_content)
            print(f"✅ {filepath} - Updated")
        else:
            print(f"⚠️  {filepath} - No NUM_ROUNDS found or different pattern")
    
    except Exception as e:
        print(f"❌ {filepath} - Error: {e}")

print("\n✨ Done!")
