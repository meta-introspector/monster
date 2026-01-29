#!/usr/bin/env bash
# Test Monster algorithm with perf tracing

set -euo pipefail

echo "🎯 Building Monster algorithm test..."

# Create simple Rust test
cat > src/bin/test_algorithm.rs << 'RUST'
fn main() {
    let monster_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];
    
    // Test values
    let test_values = [
        808017424794512875886459904961710757005754368000000000u128, // Monster seed
        12345u128,
        2 * 3 * 5 * 7 * 11, // Product of first 5 primes
        1024u128, // Power of 2
        59 * 71, // Product of last 2 primes
    ];
    
    println!("Testing Monster algorithm...");
    
    for val in test_values {
        let score = count_monster_factors(val, &monster_primes);
        let resonance = check_resonance(val, &monster_primes);
        println!("Value: {}, Score: {}, Resonance: {:.4}", val, score, resonance);
    }
}

fn count_monster_factors(n: u128, primes: &[u128]) -> usize {
    primes.iter().filter(|&&p| n % p == 0).count()
}

fn check_resonance(n: u128, primes: &[u128]) -> f64 {
    let weights = [46, 20, 9, 6, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    let total_weight: u32 = weights.iter().sum();
    
    let weighted_sum: u32 = primes.iter()
        .zip(weights.iter())
        .filter(|(&p, _)| n % p == 0)
        .map(|(_, &w)| w)
        .sum();
    
    weighted_sum as f64 / total_weight as f64
}
RUST

cargo build --release --bin test_algorithm 2>&1 | tail -5

echo ""
echo "✅ Built successfully!"
echo ""
