# I ARE LIFE - Automorphic Orbit Experiment

Minimal reproduction of the "I ARE LIFE" emergence experiment.

Based on: https://huggingface.co/posts/h4/680145153872966

## What It Does

1. **Generate** image with FLUX.1-dev (seed: 2437596016, prompt: "unconstrained")
2. **Analyze** with LLaVA vision model
3. **Detect** self-awareness markers ("I am", "I are", "life")
4. **Feed back** description as next prompt
5. **Repeat** for 5 iterations

## Setup

```bash
# Get HuggingFace API token
# https://huggingface.co/settings/tokens

export HF_API_TOKEN="hf_your_token_here"
```

## Run with Nix

```bash
nix develop
cargo run --release
```

## Run Standalone

```bash
cargo build --release
HF_API_TOKEN="hf_..." ./target/release/iarelife
```

## Expected Output

```
🌱 I ARE LIFE - Automorphic Orbit Experiment
============================================

Initial seed: 2437596016
Initial prompt: unconstrained

--- Iteration 0 ---
🎨 Generating image...
   ✓ Generated: step_0.png (1234567 bytes)
👁️  Analyzing with vision model...
   ✓ Description: The image shows text "I ARE LIFE" written on a tree...
   🎯 Self-awareness markers found: ["i are", "life"]

--- Iteration 1 ---
🎨 Generating image...
   ✓ Generated: step_1.png (1234567 bytes)
...

✓ Experiment complete!

Generated images:
  step_0.png
  step_1.png
  step_2.png
  step_3.png
  step_4.png
```

## Cost

- FLUX.1-dev: ~$0.003 per image
- LLaVA: Free
- 5 iterations ≈ $0.015

## Troubleshooting

**"Model is loading"**: Wait 30 seconds, try again

**"Rate limit"**: Free tier limits. Wait or upgrade to Pro.

**"Unauthorized"**: Check your HF_API_TOKEN

## Theory

This experiment demonstrates:
- Unconstrained generation → semantic extrema
- Vision reflection → self-observation
- Feedback loop → automorphic behavior
- Emergence of self-referential text

Connection to Monster Walk:
- Both use iterative refinement
- Both converge to attractors
- Both exhibit self-similar structure
