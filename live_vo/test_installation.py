#!/usr/bin/env python3
"""
Test script to verify that all dependencies are installed correctly
and basic functionality works.
"""

import sys

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib version: {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    return True

def test_camera():
    """Test if camera can be accessed."""
    print("\nTesting camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Camera not accessible (may be in use or not connected)")
            return False
        
        ret, frame = cap.read()
        if ret:
            print(f"✓ Camera working - frame size: {frame.shape}")
        else:
            print("✗ Could not read frame from camera")
            cap.release()
            return False
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_opencv_features():
    """Test OpenCV feature detection."""
    print("\nTesting OpenCV features...")
    
    try:
        import cv2
        import numpy as np
        
        # Create a test image
        test_img = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(test_img, (100, 100), (200, 200), 255, -1)
        cv2.circle(test_img, (400, 300), 50, 128, -1)
        
        # Test ORB detector
        orb = cv2.ORB_create(nfeatures=100)
        kp, des = orb.detectAndCompute(test_img, None)
        
        print(f"✓ ORB detector working - found {len(kp)} keypoints")
        
        # Test matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        print("✓ BFMatcher created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ OpenCV features test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Visual Odometry Demo - Installation Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test camera (optional - might not be available in some environments)
    camera_works = test_camera()
    if not camera_works:
        print("Note: Camera test failed - you may need to connect a camera or run on a different system")
    
    # Test OpenCV features
    if not test_opencv_features():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All critical tests passed!")
        print("✓ Ready to run Visual Odometry demos")
        if camera_works:
            print("✓ Camera is available - you can run the live demos")
        else:
            print("⚠ Camera not available - live demos may not work")
        
        print("\nTo run the demos:")
        print("  python live_vo.py                 # Basic VO demo")
        print("  python live_vo_with_gps.py       # VO with GPS controls")
        
    else:
        print("✗ Some tests failed - check your installation")
        print("Try: pip install opencv-python numpy matplotlib")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())