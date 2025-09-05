#!/bin/bash

echo "🎬 Welcome to the MNIST Precision Showdown Setup"
echo "------------------------------------------------"

# Step 1: Create virtual environment
echo "📦 Creating virtual environment 'gpu-lab-env'..."
python -m venv gpu-lab-env
source gpu-lab-env/bin/activate || source gpu-lab-env/Scripts/activate

# Step 2: Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Step 3: Install dependencies
echo "📥 Installing required packages..."
pip install -r requirements.txt

# Step 4: Verify GPU access
echo "🧠 Verifying CUDA availability..."
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# Step 5: Optional CuPy check
echo "🔍 Verifying CuPy GPU backend..."
python -c "import cupy; print('CuPy Device:', cupy.cuda.runtime.getDeviceCount(), 'device(s) detected')"

echo "✅ Setup complete. You're ready to run:"
echo "   python benchmark_fp32_fp16.py"
echo "   ncu --set full python kernel_inspector.py"
