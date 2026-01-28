# Running the Monster Walk AI Experiments

## Quick Start

```bash
# 1. Set your HuggingFace API token
export HF_API_TOKEN="your_token_here"

# 2. Run the experiments
cd ai-sampler
./run.sh
```

## Get HuggingFace API Token

1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it "monster-walk"
4. Select "read" access
5. Copy the token
6. Export it: `export HF_API_TOKEN="hf_..."`

## What It Does

### FLUX.1-dev (Image Generation)
- Model: `black-forest-labs/FLUX.1-dev`
- 12B parameters
- Generates images from text prompts
- Uses exact seed for reproducibility

### LLaVA (Vision Analysis)
- Model: `llava-hf/llava-1.5-7b-hf`
- Analyzes images and describes content
- Extracts text, objects, scenes

### The Loop

```
1. FLUX generates image (seed + prompt)
   ↓
2. LLaVA analyzes image
   ↓
3. Extract concepts → emoji encoding
   ↓
4. Score with model lattice (OpenCV, etc)
   ↓
5. Feed description back as next prompt
   ↓
6. Repeat until convergence
```

## Expected Output

```
🔄 Starting Automorphic Orbit with Model Lattice
=================================================
Initial prompt: unconstrained
Initial seed: 2437596016
Max iterations: 10

--- Iteration 0 ---
  🎨 Calling FLUX.1-dev API...
     Model: black-forest-labs/FLUX.1-dev
     Prompt: unconstrained
     Seed: 2437596016
     ✓ Generated 1234567 bytes
     ✓ Saved: emergence/images/step_000_seed_2437596016.png

  👁️  Calling LLaVA API...
     ✓ Analysis: The image shows text "I ARE LIFE" written on a tree...

  🔍 Extracting concepts...
     Concepts: ["life", "tree", "text", "self"]
     Emoji: 🌱🌳📝👁️

  📊 Scoring with model lattice...
    Level 0: Classical CV
      opencv-text - accuracy: 85.00%, latency: 5ms
      opencv-edge - accuracy: 60.00%, latency: 3ms
    Level 3: Large Vision
      llava-7b - accuracy: 95.00%, latency: 1200ms

  📊 Similarity to previous: 0.00%

--- Iteration 1 ---
  🎨 Calling FLUX.1-dev API...
     Prompt: reflect on: 🌱🌳📝👁️ The image shows...
     ...

  ✓ CONVERGED at iteration 4

✓ Complete! Full trace saved to emergence/orbits/
```

## Files Generated

```
emergence/
├── orbits/
│   ├── orbit_2437596016.json           # Full data
│   ├── orbit_2437596016_REPORT.md      # Human-readable
│   ├── orbit_2437596016_LATTICE.md     # Model scores
│   └── orbit_2437596016_progress.json  # Real-time updates
└── images/
    ├── step_000_seed_2437596016.png    # Generated images
    ├── step_001_seed_2437596017.png
    └── ...
```

## Troubleshooting

### "HF_API_TOKEN not set"
Export your token: `export HF_API_TOKEN="hf_..."`

### "API error: Model is loading"
HuggingFace models need to warm up. Wait 30 seconds and try again.

### "Rate limit exceeded"
Free tier has limits. Wait a few minutes or upgrade to Pro.

## Cost Estimate

- FLUX.1-dev: ~$0.003 per image
- LLaVA: Free (inference API)
- 10 iterations ≈ $0.03

## Alternative: Local Models

To run locally without API costs, use mistral.rs with GGUF models:

```bash
# Download models
wget https://huggingface.co/second-state/FLUX.1-dev-GGUF/resolve/main/flux1-dev-Q4_0.gguf
wget https://huggingface.co/llava-hf/llava-1.5-7b-hf/resolve/main/model.gguf

# Update code to use local models
# (TODO: integrate mistral.rs local inference)
```

## Next Steps

1. Run the experiments
2. View the generated images
3. Read the reports
4. Analyze convergence patterns
5. Test different seeds!

**Let's see what emerges!** 🌱✨
