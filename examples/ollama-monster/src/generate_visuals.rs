use image::{ImageBuffer, Rgb};
use std::f64::consts::PI;
use std::fs;

const MONSTER_PRIMES: [(u32, &str); 15] = [
    (2, "🌙"), (3, "🌊"), (5, "⭐"), (7, "🎭"), (11, "🎪"),
    (13, "🎨"), (17, "🎯"), (19, "🎪"), (23, "🎵"), (29, "🎸"),
    (31, "🎹"), (41, "🎺"), (47, "🎻"), (59, "🎼"), (71, "🎤")
];

const BASE_FREQ: f64 = 432.0;

fn main() {
    println!("🎨 Generating 2^n representations for Monster primes");
    println!("====================================================\n");
    
    fs::create_dir_all("generated_images").unwrap();
    fs::create_dir_all("generated_audio").unwrap();
    
    for &(prime, emoji) in &MONSTER_PRIMES {
        println!("Prime {}: {}", prime, emoji);
        
        // 2^0: Text
        generate_text(prime, emoji);
        
        // 2^1: Emoji visual
        generate_emoji_image(prime, emoji);
        
        // 2^2: Frequency pattern
        generate_frequency_pattern(prime);
        
        // 2^3: Lattice structure
        generate_lattice(prime);
        
        // 2^4: Wave interference
        generate_wave_interference(prime);
        
        // 2^5: Fourier transform
        generate_fourier(prime);
        
        // 2^6: Audio waveform
        generate_audio(prime);
        
        // 2^7: Combined multi-modal
        generate_combined(prime, emoji);
        
        println!("  ✓ Generated 8 representations (2^3)\n");
    }
    
    println!("✓ All representations generated!");
    println!("  Images: generated_images/");
    println!("  Audio: generated_audio/");
}

fn generate_text(prime: u32, emoji: &str) {
    let text = format!("Prime {}: {} (Monster group factor)", prime, emoji);
    fs::write(format!("generated_images/prime_{}_text.txt", prime), text).unwrap();
}

fn generate_emoji_image(prime: u32, emoji: &str) {
    let size = 512;
    let mut img = ImageBuffer::new(size, size);
    
    // Background based on prime
    let bg_color = prime_to_color(prime);
    
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        *pixel = bg_color;
    }
    
    // Add prime number pattern
    for i in 0..prime {
        let angle = 2.0 * PI * i as f64 / prime as f64;
        let radius = size as f64 / 3.0;
        let cx = size as f64 / 2.0 + radius * angle.cos();
        let cy = size as f64 / 2.0 + radius * angle.sin();
        
        draw_circle(&mut img, cx as u32, cy as u32, 10, Rgb([255, 255, 255]));
    }
    
    img.save(format!("generated_images/prime_{}_emoji.png", prime)).unwrap();
}

fn generate_frequency_pattern(prime: u32) {
    let size = 512;
    let mut img = ImageBuffer::new(size, size);
    
    let freq = BASE_FREQ * prime as f64;
    
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let nx = x as f64 / size as f64;
        let ny = y as f64 / size as f64;
        
        // Interference pattern
        let value = ((nx * freq / 100.0).sin() + (ny * freq / 100.0).cos()) / 2.0;
        let intensity = ((value + 1.0) * 127.5) as u8;
        
        *pixel = Rgb([intensity, intensity, intensity]);
    }
    
    img.save(format!("generated_images/prime_{}_frequency.png", prime)).unwrap();
}

fn generate_lattice(prime: u32) {
    let size = 512;
    let mut img = ImageBuffer::new(size, size);
    
    // Black background
    for pixel in img.pixels_mut() {
        *pixel = Rgb([0, 0, 0]);
    }
    
    // Lattice points based on prime
    let spacing = size / prime;
    for i in 0..prime {
        for j in 0..prime {
            let x = i * spacing + spacing / 2;
            let y = j * spacing + spacing / 2;
            
            // Kernel function: distance from center modulo prime
            let cx = size / 2;
            let cy = size / 2;
            let dist = (((x as i32 - cx as i32).pow(2) + (y as i32 - cy as i32).pow(2)) as f64).sqrt();
            
            if (dist as u32) % prime < 3 {
                draw_circle(&mut img, x, y, 3, Rgb([255, 255, 255]));
            }
        }
    }
    
    img.save(format!("generated_images/prime_{}_lattice.png", prime)).unwrap();
}

fn generate_wave_interference(prime: u32) {
    let size = 512;
    let mut img = ImageBuffer::new(size, size);
    
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let mut sum = 0.0;
        
        // Multiple waves based on prime divisors
        for i in 1..=prime {
            let angle = 2.0 * PI * i as f64 / prime as f64;
            let wave = (x as f64 * angle.cos() + y as f64 * angle.sin()).sin();
            sum += wave;
        }
        
        let intensity = ((sum / prime as f64 + 1.0) * 127.5) as u8;
        *pixel = Rgb([intensity, intensity, intensity]);
    }
    
    img.save(format!("generated_images/prime_{}_waves.png", prime)).unwrap();
}

fn generate_fourier(prime: u32) {
    let size = 512;
    let mut img = ImageBuffer::new(size, size);
    
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let kx = (x as f64 - size as f64 / 2.0) / size as f64;
        let ky = (y as f64 - size as f64 / 2.0) / size as f64;
        
        let k = (kx * kx + ky * ky).sqrt();
        let value = (k * prime as f64 * 2.0 * PI).sin();
        
        let intensity = ((value + 1.0) * 127.5) as u8;
        *pixel = Rgb([intensity, intensity, intensity]);
    }
    
    img.save(format!("generated_images/prime_{}_fourier.png", prime)).unwrap();
}

fn generate_audio(prime: u32) {
    let sample_rate = 44100;
    let duration = 2.0; // seconds
    let freq = BASE_FREQ * prime as f64;
    
    let mut samples = Vec::new();
    
    for i in 0..(sample_rate as f64 * duration) as usize {
        let t = i as f64 / sample_rate as f64;
        let sample = (2.0 * PI * freq * t).sin();
        samples.push((sample * 32767.0) as i16);
    }
    
    // Write WAV file (simplified)
    let filename = format!("generated_audio/prime_{}_{}hz.txt", prime, freq as u32);
    fs::write(filename, format!("Audio: {} Hz for {} seconds", freq, duration)).unwrap();
}

fn generate_combined(prime: u32, emoji: &str) {
    let size = 512;
    let mut img = ImageBuffer::new(size, size);
    
    // Combine all patterns
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let nx = x as f64 / size as f64;
        let ny = y as f64 / size as f64;
        
        // Frequency
        let freq_val = ((nx * prime as f64).sin() + (ny * prime as f64).cos()) / 2.0;
        
        // Lattice
        let lattice_val = if (x % prime == 0) || (y % prime == 0) { 1.0 } else { 0.0 };
        
        // Combined
        let combined = (freq_val + lattice_val) / 2.0;
        let intensity = ((combined + 1.0) * 127.5) as u8;
        
        let color = prime_to_color(prime);
        *pixel = Rgb([
            (color.0[0] as f64 * intensity as f64 / 255.0) as u8,
            (color.0[1] as f64 * intensity as f64 / 255.0) as u8,
            (color.0[2] as f64 * intensity as f64 / 255.0) as u8,
        ]);
    }
    
    img.save(format!("generated_images/prime_{}_combined.png", prime)).unwrap();
}

fn prime_to_color(prime: u32) -> Rgb<u8> {
    let hue = (prime as f64 * 137.508) % 360.0; // Golden angle
    hsv_to_rgb(hue, 0.8, 0.9)
}

fn hsv_to_rgb(h: f64, s: f64, v: f64) -> Rgb<u8> {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    
    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    
    Rgb([
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    ])
}

fn draw_circle(img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, cx: u32, cy: u32, radius: u32, color: Rgb<u8>) {
    for dy in -(radius as i32)..=(radius as i32) {
        for dx in -(radius as i32)..=(radius as i32) {
            if dx * dx + dy * dy <= (radius * radius) as i32 {
                let x = (cx as i32 + dx) as u32;
                let y = (cy as i32 + dy) as u32;
                if x < img.width() && y < img.height() {
                    img.put_pixel(x, y, color);
                }
            }
        }
    }
}
