fn main() {
    let monster_primes = [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];
    
    let test_values = [
        12345u64,
        2310u64, // 2*3*5*7*11
        1024u64, // 2^10
        4189u64, // 59*71
        30030u64, // 2*3*5*7*11*13
    ];
    
    println!("🎯 Testing Monster algorithm\n");
    
    for val in test_values {
        let score = count_monster_factors(val, &monster_primes);
        let resonance = check_resonance(val, &monster_primes);
        println!("Value: {}, Score: {}, Resonance: {:.4}", val, score, resonance);
    }
}

fn count_monster_factors(n: u64, primes: &[u64]) -> usize {
    primes.iter().filter(|&&p| n % p == 0).count()
}

fn check_resonance(n: u64, primes: &[u64]) -> f64 {
    let weights = [46, 20, 9, 6, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    let total_weight: u32 = weights.iter().sum();
    
    let weighted_sum: u32 = primes.iter()
        .zip(weights.iter())
        .filter(|(&p, _)| n % p == 0)
        .map(|(_, &w)| w)
        .sum();
    
    weighted_sum as f64 / total_weight as f64
}
