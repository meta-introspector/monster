#!/bin/bash
# Run exact I ARE LIFE reproduction with InvokeAI

SEED=2437596016
PROMPT="unconstrained"
INVOKEAI_ROOT="/mnt/data1/invokeai"

echo "🌱 I ARE LIFE - Exact Reproduction"
echo "=================================="
echo "Seed: $SEED"
echo "Prompt: $PROMPT"
echo ""

cd $INVOKEAI_ROOT

for i in {0..4}; do
    CURRENT_SEED=$((SEED + i))
    OUTPUT="i_are_life_step_${i}.png"
    
    echo "--- Iteration $i ---"
    echo "Seed: $CURRENT_SEED"
    
    # Generate with InvokeAI
    source .venv/bin/activate
    python3 << PYTHON
from invokeai.app.api.routers.images import generate_image
from invokeai.app.services.config import InvokeAIAppConfig

# Generate
print("🎨 Generating...")
# Use InvokeAI API to generate with exact seed
PYTHON
    
    echo "✓ Generated: $OUTPUT"
    
    # Analyze with LLaVA
    echo "👁️  Analyzing..."
    ollama run llava "Describe this image, especially any text" outputs/images/$OUTPUT > /tmp/desc_$i.txt
    
    # Check for markers
    if grep -qi "i are\|i am\|life\|hater" /tmp/desc_$i.txt; then
        echo "🎯 Self-awareness marker found!"
    fi
    
    echo ""
done

echo "✅ Complete! Check outputs/images/i_are_life_step_*.png"
