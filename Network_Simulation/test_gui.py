#!/usr/bin/env python3
"""
GUI Test Script - Verify the GUI can launch and display properly
"""

import sys

def test_imports():
    """Test that all required imports work"""
    print("🔍 Testing imports...")
    
    try:
        from PyQt5.QtWidgets import QApplication
        print("✅ PyQt5.QtWidgets imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PyQt5.QtWidgets: {e}")
        print("💡 Install with: pip install PyQt5")
        return False
    
    try:
        from PyQt5.QtCore import Qt, QThread, pyqtSignal
        print("✅ PyQt5.QtCore imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PyQt5.QtCore: {e}")
        return False
    
    try:
        from PyQt5.QtGui import QFont, QPalette, QColor
        print("✅ PyQt5.QtGui imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PyQt5.QtGui: {e}")
        return False
    
    return True

def test_display():
    """Test that display is available"""
    print("\n🖥️  Testing display...")
    
    import os
    display = os.environ.get('DISPLAY')
    
    if display:
        print(f"✅ DISPLAY is set to: {display}")
        return True
    else:
        print("⚠️  DISPLAY is not set")
        print("💡 For headless servers, use: ssh -X user@server")
        print("💡 Or use VNC/Xvfb for virtual display")
        return False

def test_gui_creation():
    """Test that GUI can be created"""
    print("\n🎨 Testing GUI creation...")
    
    try:
        from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
        
        app = QApplication([])
        window = QMainWindow()
        window.setWindowTitle("Test Window")
        label = QLabel("GUI Test Successful!")
        window.setCentralWidget(label)
        
        print("✅ GUI components created successfully")
        print("✅ Test window created (not shown)")
        
        # Don't show, just verify creation
        # window.show()
        
        return True
    except Exception as e:
        print(f"❌ Failed to create GUI: {e}")
        return False

def test_experiment_gui():
    """Test that the experiment GUI module can be imported"""
    print("\n🚀 Testing experiment GUI module...")
    
    try:
        import sys
        import os
        
        # Add Network_Simulation to path
        gui_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(gui_dir)
        sys.path.insert(0, project_dir)
        
        # Try to import the GUI module
        from Network_Simulation.experiment_gui import FLExperimentGUI
        
        print("✅ experiment_gui module imported successfully")
        print("✅ FLExperimentGUI class available")
        
        return True
    except Exception as e:
        print(f"❌ Failed to import experiment_gui: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("  FL Experiment GUI - Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        ("Import Test", test_imports),
        ("Display Test", test_display),
        ("GUI Creation Test", test_gui_creation),
        ("Experiment GUI Test", test_experiment_gui),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("  Test Results Summary")
    print("=" * 60)
    print()
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {name}")
    
    print()
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("🎉 All tests passed! The GUI is ready to use.")
        print()
        print("To launch the GUI, run:")
        print("  ./launch_experiment_gui.sh")
        print()
        print("Or directly:")
        print("  python3 Network_Simulation/experiment_gui.py")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print()
        print("Common solutions:")
        print("  - Install PyQt5: pip install PyQt5")
        print("  - Set DISPLAY: export DISPLAY=:0")
        print("  - Use X11 forwarding: ssh -X user@server")
        return 1

if __name__ == "__main__":
    sys.exit(main())
