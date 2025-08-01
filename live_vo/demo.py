#!/usr/bin/env python3
"""
Demo launcher script for Visual Odometry demos.
This script helps users choose and run the appropriate demo.
"""

import sys
import subprocess
import os

def show_menu():
    """Display the demo menu."""
    print("\n" + "="*60)
    print("   VISUAL ODOMETRY DEMO LAUNCHER")
    print("="*60)
    print("Available demos:")
    print()
    print("1. Basic Visual Odometry Demo")
    print("   - Live camera tracking")
    print("   - Real-time trajectory visualization")
    print("   - CSV export on exit (ESC)")
    print()
    print("2. Enhanced VO with GPS Control Stub")
    print("   - All features of basic demo")
    print("   - GPS simulation controls")
    print("   - Interactive pose clearing")
    print("   - Color-coded trajectory (Green=GPS, Red=VO-only)")
    print()
    print("3. Test Installation")
    print("   - Verify all dependencies")
    print("   - Test camera access")
    print("   - Check OpenCV features")
    print()
    print("4. View Project Structure")
    print("   - Show all created files")
    print("   - Display Android skeleton info")
    print()
    print("0. Exit")
    print("="*60)

def run_basic_vo():
    """Run the basic visual odometry demo."""
    print("\nStarting Basic Visual Odometry Demo...")
    print("Controls: ESC to exit and save trajectory.csv")
    print("Move your camera around to see trajectory tracking!")
    print()
    
    try:
        subprocess.run([sys.executable, "live_vo.py"], check=True)
    except subprocess.CalledProcessError:
        print("Error: Failed to run basic VO demo")
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")

def run_enhanced_vo():
    """Run the enhanced visual odometry demo with GPS controls."""
    print("\nStarting Enhanced Visual Odometry Demo...")
    print("Controls:")
    print("  ESC - Exit and save trajectory")
    print("  C   - Clear pose and trajectory")
    print("  S   - Stop GPS (VO-only mode, red trajectory)")
    print("  R   - Restart GPS (green trajectory)")
    print()
    
    try:
        subprocess.run([sys.executable, "live_vo_with_gps.py"], check=True)
    except subprocess.CalledProcessError:
        print("Error: Failed to run enhanced VO demo")
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")

def run_installation_test():
    """Run the installation test."""
    print("\nRunning Installation Test...")
    print()
    
    try:
        result = subprocess.run([sys.executable, "test_installation.py"], check=True)
        if result.returncode == 0:
            print("\n✓ Installation test completed successfully!")
        else:
            print("\n⚠ Installation test completed with warnings")
    except subprocess.CalledProcessError:
        print("Error: Installation test failed")
    except KeyboardInterrupt:
        print("\nTest interrupted by user")

def show_project_structure():
    """Display the project structure and information."""
    print("\nPROJECT STRUCTURE:")
    print("="*50)
    
    # Show Python files
    print("\nPython Components:")
    python_files = [
        ("live_vo.py", "Basic visual odometry demo"),
        ("live_vo_with_gps.py", "Enhanced VO with GPS controls"),
        ("test_installation.py", "Installation verification script"),
        ("demo.py", "This launcher script"),
        ("README.md", "Comprehensive documentation")
    ]
    
    for filename, description in python_files:
        status = "✓" if os.path.exists(filename) else "✗"
        print(f"  {status} {filename:<25} - {description}")
    
    # Show Android components
    print("\nAndroid ARCore Skeleton:")
    android_files = [
        ("android_arcore_gps/app/build.gradle", "App dependencies"),
        ("android_arcore_gps/app/src/main/AndroidManifest.xml", "Permissions & metadata"),
        ("android_arcore_gps/app/src/main/java/com/example/arcoregps/MainActivity.kt", "Main activity"),
        ("android_arcore_gps/app/src/main/res/layout/activity_main.xml", "UI layout"),
        ("android_arcore_gps/app/src/main/res/values/strings.xml", "String resources")
    ]
    
    for filepath, description in android_files:
        status = "✓" if os.path.exists(filepath) else "✗"
        filename = os.path.basename(filepath)
        print(f"  {status} {filename:<25} - {description}")
    
    print("\nVirtual Environment:")
    venv_status = "✓" if os.path.exists("venv/bin/activate") else "✗"
    print(f"  {venv_status} venv/                    - Python virtual environment")
    
    print("\nTo explore the Android project:")
    print("  1. Open 'android_arcore_gps/' in Android Studio")
    print("  2. Sync Gradle dependencies")
    print("  3. Connect ARCore-compatible device")
    print("  4. Build and run")
    
    print("\nGenerated Files (after running demos):")
    print("  • trajectory.csv          - Camera trajectory data")
    print("  • arcore_gps_trajectory.csv - Android combined AR+GPS data")

def main():
    """Main demo launcher."""
    while True:
        show_menu()
        
        try:
            choice = input("\nEnter your choice (0-4): ").strip()
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            sys.exit(0)
        
        if choice == "1":
            run_basic_vo()
        elif choice == "2":
            run_enhanced_vo()
        elif choice == "3":
            run_installation_test()
        elif choice == "4":
            show_project_structure()
        elif choice == "0":
            print("\nGoodbye!")
            sys.exit(0)
        else:
            print(f"\nInvalid choice: '{choice}'. Please enter 0-4.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists("live_vo.py"):
        print("Error: Please run this script from the live_vo directory")
        sys.exit(1)
    
    main()