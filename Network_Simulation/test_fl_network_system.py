#!/usr/bin/env python3
"""
FL Network Control System Test
Verifies that the network control system is properly set up and functional
"""

import subprocess
import sys
import os
from typing import Tuple

def run_command(command: list) -> Tuple[bool, str, str]:
    """Execute command and return result"""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10
        )
        return (result.returncode == 0, result.stdout, result.stderr)
    except Exception as e:
        return (False, '', str(e))

def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result"""
    status = "✓ PASS" if passed else "✗ FAIL"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status}{reset} {test_name}")
    if details and not passed:
        print(f"       {details}")

def main():
    print("=" * 70)
    print("FL Network Control System - Verification Test")
    print("=" * 70)
    print()
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Check if scripts exist
    print("[1/10] Checking if scripts exist...")
    scripts = [
        'fl_network_monitor.py',
        'fl_training_dashboard.py',
        'fl_network_control.sh'
    ]
    
    all_exist = True
    for script in scripts:
        if not os.path.exists(script):
            print_result(f"  {script}", False, "File not found")
            all_exist = False
            tests_failed += 1
        else:
            print_result(f"  {script}", True)
    
    if all_exist:
        tests_passed += 1
    
    # Test 2: Check if scripts are executable
    print("\n[2/10] Checking if scripts are executable...")
    all_executable = True
    for script in scripts:
        if os.path.exists(script) and not os.access(script, os.X_OK):
            print_result(f"  {script}", False, "Not executable")
            all_executable = False
            tests_failed += 1
        elif os.path.exists(script):
            print_result(f"  {script}", True)
    
    if all_executable:
        tests_passed += 1
    
    # Test 3: Check Docker availability
    print("\n[3/10] Checking Docker availability...")
    success, stdout, stderr = run_command(['docker', '--version'])
    print_result("Docker installed", success, stderr)
    if success:
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Test 4: Check sudo access
    print("\n[4/10] Checking sudo access...")
    success, stdout, stderr = run_command(['sudo', '-n', 'echo', 'test'])
    print_result("Sudo access", success, "May need password for tc commands")
    if success:
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Test 5: Check tc (traffic control) availability
    print("\n[5/10] Checking tc (traffic control) availability...")
    success, stdout, stderr = run_command(['tc', '-V'])
    print_result("tc command available", success, stderr)
    if success:
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Test 6: Check if Docker containers are running
    print("\n[6/10] Checking for running Docker containers...")
    success, stdout, stderr = run_command(['docker', 'ps'])
    containers = [line.strip() for line in stdout.split('\n') if 'client' in line.lower()]
    has_containers = len(containers) > 0
    print_result(f"Found {len(containers)} client container(s)", has_containers,
                "Start FL containers first" if not has_containers else "")
    if has_containers:
        tests_passed += 1
        for container_line in containers[:3]:  # Show first 3
            print(f"       {container_line[:70]}...")
    else:
        tests_failed += 1
    
    # Test 7: Test fl_network_monitor.py help
    print("\n[7/10] Testing fl_network_monitor.py...")
    success, stdout, stderr = run_command(['python3', 'fl_network_monitor.py', '--help'])
    print_result("fl_network_monitor.py --help", success, stderr)
    if success:
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Test 8: Test fl_training_dashboard.py help
    print("\n[8/10] Testing fl_training_dashboard.py...")
    success, stdout, stderr = run_command(['python3', 'fl_training_dashboard.py', '--help'])
    print_result("fl_training_dashboard.py --help", success, stderr)
    if success:
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Test 9: Test fl_network_control.sh
    print("\n[9/10] Testing fl_network_control.sh...")
    success, stdout, stderr = run_command(['./fl_network_control.sh'])
    # Script returns 1 when showing usage, which is expected behavior
    has_usage = "FL Network Control" in stdout
    print_result("fl_network_control.sh", has_usage, stderr if not has_usage else "")
    if has_usage:
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Test 10: Check Python dependencies
    print("\n[10/10] Checking Python environment...")
    success, stdout, stderr = run_command(['python3', '--version'])
    print_result("Python 3 available", success, stderr)
    if success:
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    total_tests = tests_passed + tests_failed
    print(f"Total Tests:  {total_tests}")
    print(f"Passed:       \033[92m{tests_passed}\033[0m")
    print(f"Failed:       \033[91m{tests_failed}\033[0m")
    
    if tests_failed == 0:
        print("\n\033[92m✓ All tests passed! System is ready to use.\033[0m")
        print("\nNext steps:")
        print("  1. Start FL containers: docker-compose up -d")
        print("  2. Start monitoring: python3 fl_training_dashboard.py")
        print("  3. Control network: python3 fl_network_monitor.py --monitor")
        return 0
    else:
        print("\n\033[91m✗ Some tests failed. Please fix the issues above.\033[0m")
        return 1

if __name__ == '__main__':
    sys.exit(main())
