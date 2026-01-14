#!/bin/bash
# Setup script for genomic-mechanistic-interpretability

echo "========================================="
echo "Setting up Genomic Interpretability Framework"
echo "========================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: Please run this script from the genomic-mechanistic-interpretability directory"
    exit 1
fi

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install requirements
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install requirements"
    exit 1
fi

echo ""
echo "✓ Dependencies installed successfully"

# Create output directories
echo ""
echo "Creating output directories..."
mkdir -p outputs/attention
mkdir -p outputs/ablation
mkdir -p outputs/data_exploration
mkdir -p outputs/full_analysis

echo "✓ Output directories created"

# Test imports
echo ""
echo "Testing imports..."
python -c "
import torch
import transformers
import numpy as np
import pandas as pd
import matplotlib
import seaborn
import plotly
print('✓ All required packages are importable')
"

if [ $? -ne 0 ]; then
    echo "Error: Some packages failed to import"
    exit 1
fi

# Check CUDA availability
echo ""
echo "Checking CUDA availability..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA is available')
    print(f'  Device count: {torch.cuda.device_count()}')
    print(f'  Current device: {torch.cuda.current_device()}')
    print(f'  Device name: {torch.cuda.get_device_name(0)}')
else:
    print('⚠ CUDA is not available - analyses will run on CPU (slower)')
"

# Check model path
echo ""
echo "Checking model path..."
MODEL_PATH="/project/def-mahadeva/ranaab/genomic-FM/models/dnabert2"
if [ -d "$MODEL_PATH" ]; then
    echo "✓ Model found at $MODEL_PATH"
else
    echo "⚠ Model not found at $MODEL_PATH"
    echo "  Please update MODEL_PATH in scripts if your model is elsewhere"
fi

# Check checkpoint
echo ""
echo "Checking checkpoint..."
CHECKPOINT="/scratch/ranaab/job_198160/GV-Rep/9p1e0e6n/checkpoints/epoch=99-step=18800.ckpt"
if [ -f "$CHECKPOINT" ]; then
    echo "✓ Checkpoint found at $CHECKPOINT"
else
    echo "⚠ Checkpoint not found at $CHECKPOINT"
    echo "  You may need to update CHECKPOINT_PATH in scripts"
    echo "  Note: Checkpoint might be on /scratch and only visible from compute nodes"
fi

# Print next steps
echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Run data exploration:"
echo "   python data/explore_data.py \\"
echo "       --data_dir /project/def-mahadeva/ranaab/genomic-FM/root/data \\"
echo "       --output_dir ./outputs/data_exploration"
echo ""
echo "2. Run attention analysis:"
echo "   python scripts/run_attention_analysis.py \\"
echo "       --model_path $MODEL_PATH \\"
echo "       --output_dir ./outputs/attention"
echo ""
echo "3. Run quick checkpoint analysis:"
echo "   python scripts/analyze_checkpoint.py"
echo ""
echo "4. Run full analysis pipeline:"
echo "   python scripts/run_full_analysis.py \\"
echo "       --model_path $MODEL_PATH \\"
echo "       --data_dir /project/def-mahadeva/ranaab/genomic-FM/root/data \\"
echo "       --output_dir ./outputs/full_analysis"
echo ""
echo "For more examples, see QUICK_REFERENCE.md"
echo "For detailed documentation, see README.md"
echo "For implementation details, see IMPLEMENTATION_SUMMARY.md"
echo ""
