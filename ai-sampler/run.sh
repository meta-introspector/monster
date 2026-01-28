#!/bin/bash

# Setup script for running the Monster Walk AI experiments

echo "🎪 Monster Walk AI - Setup"
echo "=========================="
echo ""

# Check for HuggingFace API token
if [ -z "$HF_API_TOKEN" ]; then
    echo "⚠️  HF_API_TOKEN not set!"
    echo ""
    echo "To use FLUX.1-dev and LLaVA, you need a HuggingFace API token:"
    echo ""
    echo "1. Go to: https://huggingface.co/settings/tokens"
    echo "2. Create a token with 'read' access"
    echo "3. Export it:"
    echo "   export HF_API_TOKEN='your_token_here'"
    echo ""
    echo "Or add to ~/.bashrc:"
    echo "   echo 'export HF_API_TOKEN=\"your_token\"' >> ~/.bashrc"
    echo ""
    exit 1
fi

echo "✓ HF_API_TOKEN found"
echo ""

# Build the project
echo "🔨 Building project..."
cargo build --release

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo "✓ Build successful"
echo ""

# Run the experiments
echo "🚀 Running experiments..."
echo ""

echo "Experiment 1: 'I ARE LIFE' (seed: 2437596016)"
cargo run --release --bin orbit-runner

echo ""
echo "✓ Complete!"
echo ""
echo "📁 Results saved to:"
echo "   emergence/orbits/"
echo "   emergence/images/"
echo ""
echo "📊 View reports:"
echo "   cat emergence/orbits/orbit_2437596016_REPORT.md"
echo "   cat emergence/orbits/orbit_2437596016_LATTICE.md"
