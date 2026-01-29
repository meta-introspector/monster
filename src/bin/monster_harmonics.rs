use std::f32::consts::PI;
use std::fs::File;
use std::io::Write;

/// The 15 Monster primes
const MONSTER_PRIMES: [u32; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];

/// The Monster prime powers
const MONSTER_POWERS: [u32; 15] = [46, 20, 9, 6, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1];

/// Base frequency (A4 = 440 Hz)
const BASE_FREQ: f32 = 440.0;

/// Sample rate (44.1 kHz)
const SAMPLE_RATE: u32 = 44100;

/// Duration in seconds
const DURATION: f32 = 10.0;

#[derive(Debug)]
struct Harmonic {
    prime: u32,
    power: u32,
    frequency: f32,
    amplitude: f32,
}

impl Harmonic {
    fn new(prime: u32, power: u32) -> Self {
        let frequency = BASE_FREQ * (prime as f32 / 2.0);
        let amplitude = 1.0 / (power as f32 + 1.0);
        Self { prime, power, frequency, amplitude }
    }
    
    fn sample(&self, t: f32) -> f32 {
        self.amplitude * (2.0 * PI * self.frequency * t).sin()
    }
}

fn generate_harmonics() -> Vec<Harmonic> {
    MONSTER_PRIMES.iter()
        .zip(MONSTER_POWERS.iter())
        .map(|(&p, &pow)| Harmonic::new(p, pow))
        .collect()
}

fn generate_waveform(harmonics: &[Harmonic]) -> Vec<f32> {
    let num_samples = (SAMPLE_RATE as f32 * DURATION) as usize;
    let mut samples = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let t = i as f32 / SAMPLE_RATE as f32;
        let sample: f32 = harmonics.iter()
            .map(|h| h.sample(t))
            .sum();
        samples.push(sample);
    }
    
    samples
}

fn normalize(samples: &mut [f32]) {
    let max_amp = samples.iter()
        .map(|s| s.abs())
        .fold(0.0f32, f32::max);
    
    if max_amp > 0.0 {
        for sample in samples.iter_mut() {
            *sample /= max_amp;
        }
    }
}

fn float_to_pcm16(f: f32) -> i16 {
    (f * 32767.0) as i16
}

fn write_wav_file(filename: &str, samples: &[f32]) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    
    let num_samples = samples.len() as u32;
    let byte_rate = SAMPLE_RATE * 2; // 16-bit mono
    let data_size = num_samples * 2;
    let file_size = data_size + 36;
    
    // RIFF header
    file.write_all(b"RIFF")?;
    file.write_all(&file_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;
    
    // fmt chunk
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?; // chunk size
    file.write_all(&1u16.to_le_bytes())?; // audio format (PCM)
    file.write_all(&1u16.to_le_bytes())?; // num channels (mono)
    file.write_all(&SAMPLE_RATE.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?; // block align
    file.write_all(&16u16.to_le_bytes())?; // bits per sample
    
    // data chunk
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;
    
    // Write samples
    for &sample in samples {
        let pcm = float_to_pcm16(sample);
        file.write_all(&pcm.to_le_bytes())?;
    }
    
    Ok(())
}

fn generate_llm_prompt(harmonics: &[Harmonic]) -> String {
    let mut prompt = String::from(
"Generate a song based on these Monster Walk harmonics:

Frequencies (Hz):
");
    
    for h in harmonics {
        prompt.push_str(&format!(
            "Prime {}: {:.2} Hz (amplitude {:.4})\n",
            h.prime, h.frequency, h.amplitude
        ));
    }
    
    prompt.push_str("
The song should:
1. Use these exact frequencies as the harmonic series
2. Create a melody that walks through the primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71
3. Emphasize 71 (highest frequency) as the climax
4. Use 2^46 (lowest amplitude) as the bass foundation
5. Create a sense of ascending through the Monster's structure
6. Duration: 10 seconds
7. Style: Mathematical, ethereal, building to a peak at 71

The harmonics represent the Monster group's prime factorization:
2^46 × 3^20 × 5^9 × 7^6 × 11^2 × 13^3 × 17 × 19 × 23 × 29 × 31 × 41 × 47 × 59 × 71

Make it sound like walking up a mathematical staircase to the gravity well at 71.

Generate:
1. Lyrics that mention each prime
2. Melody notes (MIDI or frequency)
3. Rhythm pattern
4. Arrangement suggestions
");
    
    prompt
}

fn main() -> std::io::Result<()> {
    println!("🎵 Monster Walk Harmonics Generator");
    println!("====================================");
    println!();
    
    // Generate harmonics
    println!("📊 Generating harmonics from Monster primes...");
    let harmonics = generate_harmonics();
    
    println!("Found {} harmonics:", harmonics.len());
    for h in &harmonics {
        println!("  Prime {}: {:.2} Hz (amplitude {:.4})", 
            h.prime, h.frequency, h.amplitude);
    }
    println!();
    
    // Generate waveform
    println!("🌊 Generating waveform...");
    let mut waveform = generate_waveform(&harmonics);
    println!("Generated {} samples", waveform.len());
    println!();
    
    // Normalize
    println!("📏 Normalizing...");
    normalize(&mut waveform);
    println!("✓ Normalized to [-1.0, 1.0]");
    println!();
    
    // Write WAV file
    println!("💾 Writing WAV file...");
    let wav_filename = "monster_walk.wav";
    write_wav_file(wav_filename, &waveform)?;
    println!("✓ Wrote {}", wav_filename);
    println!();
    
    // Generate LLM prompt
    println!("🤖 Generating LLM prompt...");
    let prompt = generate_llm_prompt(&harmonics);
    let prompt_filename = "monster_walk_prompt.txt";
    std::fs::write(prompt_filename, &prompt)?;
    println!("✓ Wrote {}", prompt_filename);
    println!();
    
    // Summary
    println!("✅ Complete!");
    println!();
    println!("Generated files:");
    println!("  - {}: 10-second audio of Monster harmonics", wav_filename);
    println!("  - {}: LLM prompt for song generation", prompt_filename);
    println!();
    println!("🎵 The Monster walks from 2^46 to 71! 🎵");
    
    Ok(())
}
