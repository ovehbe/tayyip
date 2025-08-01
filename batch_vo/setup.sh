#!/bin/bash
# Setup script for Batch Visual Odometry

echo "🚀 Setting up Batch Visual Odometry environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "📋 To use the batch VO processor:"
echo "   1. Activate the environment: source venv/bin/activate"
echo "   2. Run the demo: python demo_batch_vo.py"
echo "   3. Or use directly: python batch_visual_odometry.py --help"
echo ""
echo "🧪 To test the installation:"
echo "   source venv/bin/activate && python test_workflow.py"