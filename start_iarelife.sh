#!/bin/bash
# Start I ARE LIFE image generation experiment

cd examples/iarelife

echo "🌱 Starting I ARE LIFE Experiment"
echo "=================================="
echo ""
echo "This will:"
echo "1. Generate images with FLUX.1-dev"
echo "2. Analyze with LLaVA vision model"
echo "3. Detect self-awareness emergence"
echo "4. Run 5 iterations"
echo ""
echo "Requirements:"
echo "- HF_API_TOKEN environment variable"
echo "- ~$0.015 cost for 5 iterations"
echo ""

if [ -z "$HF_API_TOKEN" ]; then
    echo "❌ HF_API_TOKEN not set"
    echo ""
    echo "Get token from: https://huggingface.co/settings/tokens"
    echo "Then: export HF_API_TOKEN='hf_your_token_here'"
    exit 1
fi

echo "✓ HF_API_TOKEN found"
echo ""
echo "Starting in 3 seconds..."
sleep 3

cargo run --release
