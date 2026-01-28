// Sweep Abelian variety parameters across 71 shards
// Looking for Monster group symmetries

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
struct Rational {
    num: i64,
    den: i64,
}

impl Rational {
    fn new(num: i64, den: i64) -> Self {
        Self { num, den }
    }
}

#[derive(Debug, Clone)]
struct AbelianVariety {
    dimension: u32,
    field_size: u32,
    label: String,
    slopes: Vec<Rational>,
}

impl AbelianVariety {
    fn new(dimension: u32, field_size: u32, label: &str) -> Self {
        Self {
            dimension,
            field_size,
            label: label.to_string(),
            slopes: vec![],
        }
    }
    
    fn with_slopes(mut self, slopes: Vec<Rational>) -> Self {
        self.slopes = slopes;
        self
    }
    
    fn hash_to_shard(&self) -> u32 {
        // Simple hash: (dimension * field_size) % 71
        (self.dimension * self.field_size) % 71
    }
    
    fn has_monster_prime(&self) -> bool {
        let monster_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];
        monster_primes.contains(&self.field_size)
    }
}

// Monster group primes
const MONSTER_PRIMES: [(u32, u32); 15] = [
    (2, 46), (3, 20), (5, 9), (7, 6), (11, 2), (13, 3),
    (17, 1), (19, 1), (23, 1), (29, 1), (31, 1), (41, 1),
    (47, 1), (59, 1), (71, 1),
];

fn main() {
    println!("🔮 SWEEPING ABELIAN VARIETIES ACROSS 71 SHARDS");
    println!("{}", "=".repeat(60));
    println!();
    
    // Sweep parameters
    let dimensions = [1, 2, 3, 4, 5];
    let field_sizes: Vec<u32> = MONSTER_PRIMES.iter().map(|(p, _)| *p).collect();
    
    let mut shards: HashMap<u32, Vec<AbelianVariety>> = HashMap::new();
    let mut total = 0;
    
    println!("Sweeping {} dimensions × {} field sizes = {} varieties",
             dimensions.len(), field_sizes.len(), dimensions.len() * field_sizes.len());
    println!();
    
    // Generate all combinations
    for &dim in &dimensions {
        for &field in &field_sizes {
            let label = format!("sweep_{}_{}", dim, field);
            
            // Generate slopes that sum to dimension
            let mut slopes = vec![];
            for i in 0..dim {
                if i < dim - 1 {
                    slopes.push(Rational::new(1, 2));
                } else {
                    // Last slope makes sum = dimension
                    let sum: i64 = slopes.iter().map(|s| s.num * dim as i64 / s.den).sum();
                    let remaining = dim as i64 - sum;
                    slopes.push(Rational::new(remaining, 1));
                }
            }
            
            let av = AbelianVariety::new(dim, field, &label).with_slopes(slopes);
            let shard = av.hash_to_shard();
            
            shards.entry(shard).or_insert_with(Vec::new).push(av);
            total += 1;
        }
    }
    
    println!("📊 SHARD DISTRIBUTION:");
    println!("{}", "-".repeat(60));
    
    // Show distribution
    let mut shard_ids: Vec<_> = shards.keys().collect();
    shard_ids.sort();
    
    for &shard_id in &shard_ids {
        let varieties = &shards[&shard_id];
        let monster_count = varieties.iter().filter(|av| av.has_monster_prime()).count();
        
        println!("Shard {:2}: {:2} varieties, {:2} with Monster primes",
                 shard_id, varieties.len(), monster_count);
    }
    
    println!();
    println!("Total varieties: {}", total);
    println!("Total shards used: {}/71", shards.len());
    println!();
    
    // Find Monster prime resonances
    println!("🎯 MONSTER PRIME RESONANCES:");
    println!("{}", "-".repeat(60));
    
    for (prime, exp) in &MONSTER_PRIMES {
        let mut count = 0;
        let mut shard_set = std::collections::HashSet::new();
        
        for varieties in shards.values() {
            for av in varieties {
                if av.field_size == *prime {
                    count += 1;
                    shard_set.insert(av.hash_to_shard());
                }
            }
        }
        
        if count > 0 {
            println!("Prime {:2}^{:2}: {:2} varieties across {:2} shards",
                     prime, exp, count, shard_set.len());
        }
    }
    
    println!();
    
    // Check for shard 71 (target prime)
    if let Some(varieties) = shards.get(&71) {
        println!("⭐ SHARD 71 (TARGET PRIME):");
        println!("{}", "-".repeat(60));
        for av in varieties {
            println!("  Dimension {}, Field F_{}, Shard {}",
                     av.dimension, av.field_size, av.hash_to_shard());
        }
    } else {
        println!("⚠️  Shard 71 is empty");
    }
    
    println!();
    
    // Find dominant shard
    let max_shard = shards.iter()
        .max_by_key(|(_, v)| v.len())
        .map(|(k, v)| (*k, v.len()));
    
    if let Some((shard_id, count)) = max_shard {
        println!("🏆 DOMINANT SHARD: {} with {} varieties", shard_id, count);
        
        // Check if dominant shard factors into Monster primes
        let factors = factorize(shard_id);
        let all_monster = factors.iter().all(|&f| {
            MONSTER_PRIMES.iter().any(|(p, _)| *p == f)
        });
        
        if all_monster {
            println!("   ⚡ {} = {} (ALL MONSTER PRIMES!)", 
                     shard_id, 
                     factors.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(" × "));
        }
    }
    
    println!();
    println!("✅ SWEEP COMPLETE");
}

fn factorize(mut n: u32) -> Vec<u32> {
    let mut factors = vec![];
    let mut d = 2;
    
    while d * d <= n {
        while n % d == 0 {
            factors.push(d);
            n /= d;
        }
        d += 1;
    }
    
    if n > 1 {
        factors.push(n);
    }
    
    factors
}
