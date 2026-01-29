use anyhow::Result;
use serde::Serialize;
use std::fs;

const MONSTER_PRIMES: [u32; 5] = [2, 3, 5, 7, 11];

#[derive(Debug, Serialize, Clone)]
struct MultiModalPrime {
    prime: u32,
    representations: Vec<Representation>,
    transitions: Vec<Transition>,
}

#[derive(Debug, Serialize, Clone)]
struct Representation {
    mode: String,
    encoding: String,
    dimension: usize,
}

#[derive(Debug, Serialize, Clone)]
struct Transition {
    from_mode: String,
    to_mode: String,
    preserved_info: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎨 Multi-Modal Prime Representations (2^n encodings)");
    println!("===================================================\n");
    
    let mut all_primes = Vec::new();
    
    for prime in MONSTER_PRIMES {
        println!("Prime {}: Generating 2^n representations...", prime);
        
        let multi = generate_multimodal(prime).await?;
        
        println!("  Representations: {}", multi.representations.len());
        for rep in &multi.representations {
            println!("    {} (dim {}): {}", rep.mode, rep.dimension, 
                     &rep.encoding[..rep.encoding.len().min(40)]);
        }
        
        println!("  Transitions: {}", multi.transitions.len());
        for trans in &multi.transitions {
            println!("    {} → {}: {:.1}% preserved",
                     trans.from_mode, trans.to_mode, trans.preserved_info * 100.0);
        }
        
        all_primes.push(multi);
        println!();
    }
    
    // Test alternating between modes
    println!("\n🔄 Testing Mode Alternation:");
    test_alternation(&all_primes[0]).await?;
    
    fs::write("MULTIMODAL_PRIMES.json", serde_json::to_string_pretty(&all_primes)?)?;
    
    println!("\n✓ Complete! Results: MULTIMODAL_PRIMES.json");
    
    Ok(())
}

async fn generate_multimodal(prime: u32) -> Result<MultiModalPrime> {
    let mut representations = Vec::new();
    
    // 1. Text (2^0 = 1D)
    representations.push(Representation {
        mode: "text".to_string(),
        encoding: format!("Prime number {}", prime),
        dimension: 1,
    });
    
    // 2. Emoji (2^1 = 2D visual)
    let emoji = match prime {
        2 => "🌙",
        3 => "🌊",
        5 => "⭐",
        7 => "🎭",
        11 => "🎪",
        _ => "❓",
    };
    representations.push(Representation {
        mode: "emoji".to_string(),
        encoding: emoji.to_string(),
        dimension: 2,
    });
    
    // 3. Color (2^2 = 4D RGBA)
    let color = prime_to_color(prime);
    representations.push(Representation {
        mode: "color".to_string(),
        encoding: color,
        dimension: 4,
    });
    
    // 4. Frequency (2^3 = 8D harmonic)
    let freq = 432.0 * prime as f64;
    representations.push(Representation {
        mode: "frequency".to_string(),
        encoding: format!("{} Hz", freq),
        dimension: 8,
    });
    
    // 5. Geometry (2^4 = 16D shape)
    let geometry = prime_to_geometry(prime);
    representations.push(Representation {
        mode: "geometry".to_string(),
        encoding: geometry,
        dimension: 16,
    });
    
    // 6. LMFDB (2^5 = 32D database)
    let lmfdb = format!("https://www.lmfdb.org/NumberField/{}", prime);
    representations.push(Representation {
        mode: "lmfdb".to_string(),
        encoding: lmfdb,
        dimension: 32,
    });
    
    // 7. Binary (2^6 = 64D bit pattern)
    representations.push(Representation {
        mode: "binary".to_string(),
        encoding: format!("{:b}", prime),
        dimension: 64,
    });
    
    // 8. Lattice (2^7 = 128D vector space)
    representations.push(Representation {
        mode: "lattice".to_string(),
        encoding: format!("Z^{} lattice point", prime),
        dimension: 128,
    });
    
    // Calculate transitions
    let mut transitions = Vec::new();
    for i in 0..representations.len() {
        for j in 0..representations.len() {
            if i != j {
                let preserved = calculate_preservation(&representations[i], &representations[j], prime);
                transitions.push(Transition {
                    from_mode: representations[i].mode.clone(),
                    to_mode: representations[j].mode.clone(),
                    preserved_info: preserved,
                });
            }
        }
    }
    
    Ok(MultiModalPrime {
        prime,
        representations,
        transitions,
    })
}

fn prime_to_color(prime: u32) -> String {
    // Map prime to RGB color
    let r = (prime * 37) % 256;
    let g = (prime * 73) % 256;
    let b = (prime * 109) % 256;
    format!("rgb({}, {}, {})", r, g, b)
}

fn prime_to_geometry(prime: u32) -> String {
    match prime {
        2 => "line (1D)".to_string(),
        3 => "triangle (2D)".to_string(),
        5 => "pentagon (2D)".to_string(),
        7 => "heptagon (2D)".to_string(),
        11 => "hendecagon (2D)".to_string(),
        _ => format!("{}-gon", prime),
    }
}

fn calculate_preservation(from: &Representation, to: &Representation, prime: u32) -> f64 {
    // Information preserved in transition
    let dim_ratio = (from.dimension.min(to.dimension) as f64) / 
                    (from.dimension.max(to.dimension) as f64);
    
    // Check if prime number is preserved in encoding
    let prime_preserved = if to.encoding.contains(&prime.to_string()) {
        1.0
    } else {
        0.5
    };
    
    (dim_ratio + prime_preserved) / 2.0
}

async fn test_alternation(multi: &MultiModalPrime) -> Result<()> {
    println!("  Testing alternation for prime {}:", multi.prime);
    
    // Alternate: text → emoji → color → frequency → geometry
    let sequence = vec!["text", "emoji", "color", "frequency", "geometry"];
    
    for i in 0..sequence.len() - 1 {
        let from = sequence[i];
        let to = sequence[i + 1];
        
        let from_rep = multi.representations.iter()
            .find(|r| r.mode == from)
            .unwrap();
        let to_rep = multi.representations.iter()
            .find(|r| r.mode == to)
            .unwrap();
        
        let trans = multi.transitions.iter()
            .find(|t| t.from_mode == from && t.to_mode == to)
            .unwrap();
        
        println!("    {} → {}: {} → {} ({:.1}% preserved)",
                 from, to,
                 &from_rep.encoding[..from_rep.encoding.len().min(20)],
                 &to_rep.encoding[..to_rep.encoding.len().min(20)],
                 trans.preserved_info * 100.0);
    }
    
    Ok(())
}
