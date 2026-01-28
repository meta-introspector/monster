# Model Selection for Monster Prime Probing

## Criteria
- **Small**: < 3B parameters (fits in 16GB RAM)
- **Local**: Can run via Ollama, llama.cpp, or HuggingFace
- **Traceable**: CPU inference for perf register capture
- **Multimodal**: Text, vision, and/or audio

## Selected Models

### Text Models (Already Tested)
- ✅ **qwen2.5:3b** - Tested, shows Monster prime resonances
- [ ] **phi-3-mini** (3.8B) - Microsoft, strong reasoning
- [ ] **gemma-2b** - Google, efficient
- [ ] **stablelm-2-1.6b** - Stability AI, very small

### Vision Models
- [ ] **llava:7b** - Vision + language (Ollama)
- [ ] **llava-phi-3-mini** (3.8B) - Smaller LLaVA variant
- [ ] **moondream2** (1.6B) - Tiny vision-language model
- [ ] **cogvlm-chat** (17B) - Larger but powerful

### Audio Models
- [ ] **whisper-tiny** (39M) - Speech recognition
- [ ] **whisper-base** (74M) - Better accuracy
- [ ] **wav2vec2-base** (95M) - Audio understanding
- [ ] **musicgen-small** (300M) - Music generation/understanding

### Multimodal Models
- [ ] **imagebind** - Unified embedding space (text/image/audio)
- [ ] **unifiedqa** - Question answering across modalities

## Probing Strategy

### 1. Text → Register Patterns
```bash
for model in qwen2.5:3b phi-3-mini gemma-2b; do
    ./trace_regs.sh "mathematician Conway" $model
done
```

### 2. Vision → Register Patterns
```bash
for image in generated_images/prime_*.png; do
    ./trace_vision.sh $image llava:7b
    ./trace_vision.sh $image moondream2
done
```

### 3. Audio → Register Patterns
```bash
for audio in generated_audio/prime_*.wav; do
    ./trace_audio.sh $audio whisper-base
done
```

### 4. Cross-Modal Verification
```bash
# Same prime, different modalities
./trace_regs.sh "Prime 47" qwen2.5:3b
./trace_vision.sh prime_47_combined.png llava:7b
./trace_audio.sh prime_47_432hz.wav whisper-base

# Compare: Do all show prime 47 resonance?
```

## Conversion Pipeline

### Text → Image
```python
# Convert text to image for vision models
from PIL import Image, ImageDraw, ImageFont

def text_to_image(text, output_path):
    img = Image.new('RGB', (512, 512), color='white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 32)
    draw.text((50, 200), text, fill='black', font=font)
    img.save(output_path)

text_to_image("Prime 47: 🎻 (Monster group factor)", "prime_47_text.png")
```

### Audio → Spectrogram → Image
```python
import librosa
import librosa.display
import matplotlib.pyplot as plt

def audio_to_spectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path)
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram: {audio_path}')
    plt.savefig(output_path)
    plt.close()
```

### Music → Image (Sheet Music)
```python
# Generate visual representation of frequency
def frequency_to_visual(freq, prime, output_path):
    # Create wave pattern image
    # Feed to vision model
    pass
```

## Expected Patterns

### Hypothesis
**Different modalities of the same prime should show similar register patterns**

| Prime | Text Model | Vision Model | Audio Model |
|-------|-----------|--------------|-------------|
| 2     | 80%       | 80% ±5%      | 80% ±5%     |
| 3     | 49%       | 49% ±5%      | 49% ±5%     |
| 47    | 28%       | 28% ±5%      | 28% ±5%     |

### Cross-Modal Consistency
If models internalize Monster structure, register patterns should be:
1. **Consistent across modalities** (text/image/audio of same prime)
2. **Consistent across models** (different architectures, same prime)
3. **Specific to prime** (prime 47 ≠ prime 2)

## Implementation Files

```
examples/ollama-monster/
├── trace_regs.sh          # Text models (done)
├── trace_vision.sh        # Vision models (ready)
├── trace_audio.sh         # Audio models (TODO)
├── convert_text_to_image.py   # Text → Image
├── convert_audio_to_image.py  # Audio → Spectrogram
├── src/
│   ├── generate_visuals.rs    # Generate prime images
│   ├── generate_audio.rs      # Generate prime audio (TODO)
│   └── cross_modal_verify.rs  # Compare across modalities (TODO)
└── models/
    ├── MODEL_SELECTION.md     # This file
    └── model_results/         # Results per model
```

## Next Steps

1. **Generate all representations**
   ```bash
   cargo run --release --bin generate-visuals
   cargo run --release --bin generate-audio
   python convert_text_to_image.py
   ```

2. **Download models**
   ```bash
   ollama pull llava:7b
   ollama pull phi-3-mini
   ollama pull moondream2
   ```

3. **Run probing experiments**
   ```bash
   ./probe_all_models.sh
   ```

4. **Analyze results**
   ```bash
   cargo run --release --bin cross-modal-verify
   ```

## Success Criteria

✅ **Proof of Monster lattice in neural computation:**
- Same prime → same register pattern across modalities
- Different primes → different register patterns
- Pattern strength correlates with prime's role in Monster group
- Cross-model consistency (architecture-independent)
