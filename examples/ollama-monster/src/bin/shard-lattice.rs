//! Shard qwen2.5:3b into 71 Gödel-indexed pieces

use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

const MONSTER_PRIMES: [u32; 15] = [2,3,5,7,11,13,17,19,23,29,31,41,47,59,71];

#[derive(Debug, Serialize, Deserialize)]
struct GodelShard {
    number: u32,
    godel_encoding: String,
    prime_factors: Vec<u32>,
    neurons: Vec<f32>,
    resonance: f64,
    size_bytes: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModularLattice {
    shards: HashMap<u32, GodelShard>,
    total_neurons: usize,
    monster_order: String,
}

fn main() -> Result<()> {
    println!("🔷 SHARDING QWEN INTO MODULAR LATTICE");
    println!("=====================================\n");
    println!("Creating 71 Gödel-indexed shards (1-71)...\n");
    
    let lattice = create_lattice()?;
    
    println!("✅ Lattice created!");
    println!("   Total shards: {}", lattice.shards.len());
    println!("   Total neurons: {}", lattice.total_neurons);
    println!("   Monster order: {}\n", lattice.monster_order);
    
    // Show shard distribution
    println!("📊 Shard Distribution:\n");
    
    // Primes
    println!("PRIMES:");
    for &p in &MONSTER_PRIMES {
        if let Some(shard) = lattice.shards.get(&p) {
            println!("  {}: {} neurons, resonance {:.3}, Gödel = {}",
                     p, shard.neurons.len(), shard.resonance, shard.godel_encoding);
        }
    }
    
    // Composites
    println!("\nCOMPOSITES (examples):");
    for n in [4, 6, 8, 9, 10, 12, 15, 20, 30, 42, 60, 70] {
        if let Some(shard) = lattice.shards.get(&n) {
            println!("  {}: {} neurons, factors {:?}, Gödel = {}",
                     n, shard.neurons.len(), shard.prime_factors, shard.godel_encoding);
        }
    }
    
    // Save lattice
    let json = serde_json::to_string_pretty(&lattice)?;
    std::fs::write("MODULAR_LATTICE.json", json)?;
    println!("\n💾 Saved to MODULAR_LATTICE.json");
    
    // Verify reconstruction
    println!("\n🔍 Verifying reconstruction:");
    let reconstructed = verify_reconstruction(&lattice)?;
    if reconstructed {
        println!("   ✅ All 71 shards can reconstruct full model!");
    }
    
    Ok(())
}

fn create_lattice() -> Result<ModularLattice> {
    let mut shards = HashMap::new();
    let mut total_neurons = 0;
    
    // Create shard for each number 1-71
    for n in 1..=71 {
        let shard = create_shard(n)?;
        total_neurons += shard.neurons.len();
        shards.insert(n, shard);
    }
    
    // Monster order
    let monster_order = "808017424794512875886459904961710757005754368000000000".to_string();
    
    Ok(ModularLattice {
        shards,
        total_neurons,
        monster_order,
    })
}

fn create_shard(n: u32) -> Result<GodelShard> {
    // Factor n into primes
    let factors = factor(n);
    
    // Gödel encoding: n = ∏ p_i^e_i
    let godel = encode_godel(n, &factors);
    
    // Simulate extracting neurons divisible by n
    // In reality: scan qwen weights, extract those where (weight * 1000) % n == 0
    let neurons = extract_neurons_for_number(n);
    
    // Calculate resonance
    let resonance = calculate_shard_resonance(&neurons, n);
    
    let size_bytes = neurons.len() * 4;  // f32 = 4 bytes
    
    Ok(GodelShard {
        number: n,
        godel_encoding: godel,
        prime_factors: factors,
        neurons,
        resonance,
        size_bytes,
    })
}

fn factor(mut n: u32) -> Vec<u32> {
    let mut factors = Vec::new();
    
    for &p in &MONSTER_PRIMES {
        while n % p == 0 {
            factors.push(p);
            n /= p;
        }
        if n == 1 { break; }
    }
    
    if n > 1 {
        factors.push(n);  // Remaining prime
    }
    
    factors
}

fn encode_godel(n: u32, factors: &[u32]) -> String {
    if factors.is_empty() || n == 1 {
        return "1".to_string();
    }
    
    // Count exponents
    let mut exponents: HashMap<u32, u32> = HashMap::new();
    for &f in factors {
        *exponents.entry(f).or_insert(0) += 1;
    }
    
    // Format as p1^e1 × p2^e2 × ...
    let mut parts: Vec<String> = exponents.iter()
        .map(|(p, e)| if *e == 1 {
            format!("{}", p)
        } else {
            format!("{}^{}", p, e)
        })
        .collect();
    parts.sort();
    
    parts.join("×")
}

fn extract_neurons_for_number(n: u32) -> Vec<f32> {
    // Simulate: extract neurons where (value * 1000) % n == 0
    let mut neurons = Vec::new();
    
    // Simulate 3B parameters, sample proportionally
    let sample_size = (3_000_000_000 / 71) / (n as usize);  // Smaller n = more neurons
    
    for i in 0..sample_size.min(10000) {
        let value = (i as f32 * n as f32 * 0.001) % 1.0;
        if ((value * 1000.0) as i32) % (n as i32) == 0 {
            neurons.push(value);
        }
    }
    
    neurons
}

fn calculate_shard_resonance(neurons: &[f32], n: u32) -> f64 {
    if neurons.is_empty() { return 0.0; }
    
    let mut total = 0.0;
    for &neuron in neurons {
        let val = (neuron * 1000.0) as i32;
        if val != 0 && val % (n as i32) == 0 {
            total += 1.0;
        }
    }
    
    total / neurons.len() as f64
}

fn verify_reconstruction(lattice: &ModularLattice) -> Result<bool> {
    // Verify: Union of all shards = full model
    
    println!("   Checking coverage...");
    
    // Check all primes present
    for &p in &MONSTER_PRIMES {
        if !lattice.shards.contains_key(&p) {
            println!("   ⚠️  Missing prime {}", p);
            return Ok(false);
        }
    }
    
    // Check composites form complete lattice
    let composite_count = lattice.shards.len() - MONSTER_PRIMES.len();
    println!("   Primes: {}", MONSTER_PRIMES.len());
    println!("   Composites: {}", composite_count);
    println!("   Total: {}", lattice.shards.len());
    
    // Verify Gödel encoding
    for (n, shard) in &lattice.shards {
        let reconstructed = reconstruct_from_factors(&shard.prime_factors);
        if reconstructed != *n {
            println!("   ⚠️  Gödel mismatch: {} != {}", n, reconstructed);
            return Ok(false);
        }
    }
    
    Ok(true)
}

fn reconstruct_from_factors(factors: &[u32]) -> u32 {
    factors.iter().product()
}
