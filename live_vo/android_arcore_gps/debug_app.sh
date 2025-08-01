#!/bin/bash

echo "🔍 ARCore GPS App Debug Helper"
echo "==============================="

# Check if device is connected
echo "📱 Checking device connection..."
DEVICES=$(adb devices | grep -v "List of devices" | grep "device" | wc -l)
if [ $DEVICES -eq 0 ]; then
    echo "❌ No devices connected!"
    exit 1
else
    echo "✅ Device connected: $(adb devices | grep device | awk '{print $1}')"
fi

# Check if app is installed
echo ""
echo "📦 Checking app installation..."
if adb shell pm list packages | grep -q "com.example.arcoregps"; then
    echo "✅ ARCore GPS Demo app is installed"
else
    echo "❌ App not installed. Installing now..."
    adb install app/build/outputs/apk/debug/app-debug.apk
fi

# Launch the app
echo ""
echo "🚀 Launching ARCore GPS Demo..."
adb shell am start -n com.example.arcoregps/.MainActivity

# Show live logs
echo ""
echo "📋 Live app logs (Ctrl+C to stop):"
echo "Look for 'CameraManager' tags to see camera debug info"
echo ""
adb logcat | grep -E "(CameraManager|arcoregps|AndroidRuntime)"