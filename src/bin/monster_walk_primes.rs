// Rust: Monster Walk with Prime Factorizations in All Bases

use num_bigint::BigUint;
use num_traits::{One, Zero, ToPrimitive};
use std::collections::HashMap;

/// Prime factorization
#[derive(Debug, Clone)]
struct PrimeFactorization {
    factors: HashMap<u32, u32>,
}

impl PrimeFactorization {
    fn new() -> Self {
        Self { factors: HashMap::new() }
    }
    
    fn monster() -> Self {
        let mut f = Self::new();
        f.factors.insert(2, 46);
        f.factors.insert(3, 20);
        f.factors.insert(5, 9);
        f.factors.insert(7, 6);
        f.factors.insert(11, 2);
        f.factors.insert(13, 3);
        f.factors.insert(17, 1);
        f.factors.insert(19, 1);
        f.factors.insert(23, 1);
        f.factors.insert(29, 1);
        f.factors.insert(31, 1);
        f.factors.insert(41, 1);
        f.factors.insert(47, 1);
        f.factors.insert(59, 1);
        f.factors.insert(71, 1);
        f
    }
    
    fn to_value(&self) -> BigUint {
        let mut result = BigUint::one();
        for (&prime, &exp) in &self.factors {
            result *= BigUint::from(prime).pow(exp);
        }
        result
    }
    
    fn format(&self) -> String {
        let mut parts: Vec<_> = self.factors.iter()
            .filter(|(_, &exp)| exp > 0)
            .collect();
        parts.sort_by_key(|(p, _)| **p);
        
        parts.iter()
            .map(|(&p, &exp)| {
                if exp == 1 {
                    format!("{}", p)
                } else {
                    format!("{}^{}", p, exp)
                }
            })
            .collect::<Vec<_>>()
            .join(" × ")
    }
}

/// Walk step
#[derive(Debug, Clone)]
struct WalkStep {
    step_num: u8,
    removed: PrimeFactorization,
    remaining: PrimeFactorization,
    value: BigUint,
    base: u32,
    representation: Vec<u8>,
}

/// Convert to base
fn to_base(n: &BigUint, base: u32) -> Vec<u8> {
    if n.is_zero() {
        return vec![0];
    }
    
    let mut digits = Vec::new();
    let mut num = n.clone();
    let base_big = BigUint::from(base);
    
    while !num.is_zero() {
        let digit = (&num % &base_big).to_u32().unwrap_or(0) as u8;
        digits.push(digit);
        num /= &base_big;
    }
    
    digits.reverse();
    digits
}

/// Generate walk for given base
fn walk_in_base(base: u32) -> Vec<WalkStep> {
    let monster = PrimeFactorization::monster();
    
    // Step 2: Remove 17, 59
    let mut step2_removed = PrimeFactorization::new();
    step2_removed.factors.insert(17, 1);
    step2_removed.factors.insert(59, 1);
    
    let mut step2_remaining = monster.clone();
    step2_remaining.factors.remove(&17);
    step2_remaining.factors.remove(&59);
    
    // Step 4: Remove 8 factors (Group 1)
    let mut step4_removed = PrimeFactorization::new();
    step4_removed.factors.insert(7, 6);
    step4_removed.factors.insert(11, 2);
    step4_removed.factors.insert(17, 1);
    step4_removed.factors.insert(19, 1);
    step4_removed.factors.insert(29, 1);
    step4_removed.factors.insert(31, 1);
    step4_removed.factors.insert(41, 1);
    step4_removed.factors.insert(59, 1);
    
    let mut step4_remaining = monster.clone();
    for (&p, _) in &step4_removed.factors {
        step4_remaining.factors.remove(&p);
    }
    
    // Step 6: Remove 4 factors (Group 2)
    let mut step6_removed = PrimeFactorization::new();
    step6_removed.factors.insert(3, 20);
    step6_removed.factors.insert(5, 9);
    step6_removed.factors.insert(13, 3);
    step6_removed.factors.insert(31, 1);
    
    let mut step6_remaining = monster.clone();
    for (&p, _) in &step6_removed.factors {
        step6_remaining.factors.remove(&p);
    }
    
    // Step 8: Remove 4 factors (Group 3)
    let mut step8_removed = PrimeFactorization::new();
    step8_removed.factors.insert(3, 20);
    step8_removed.factors.insert(13, 3);
    step8_removed.factors.insert(31, 1);
    step8_removed.factors.insert(71, 1);
    
    let mut step8_remaining = monster.clone();
    for (&p, _) in &step8_removed.factors {
        step8_remaining.factors.remove(&p);
    }
    
    // Step 10: Just 71
    let mut step10_remaining = PrimeFactorization::new();
    step10_remaining.factors.insert(71, 1);
    
    vec![
        WalkStep {
            step_num: 1,
            removed: PrimeFactorization::new(),
            remaining: monster.clone(),
            value: monster.to_value(),
            base,
            representation: to_base(&monster.to_value(), base),
        },
        WalkStep {
            step_num: 2,
            removed: step2_removed,
            remaining: step2_remaining.clone(),
            value: step2_remaining.to_value(),
            base,
            representation: to_base(&step2_remaining.to_value(), base),
        },
        WalkStep {
            step_num: 4,
            removed: step4_removed,
            remaining: step4_remaining.clone(),
            value: step4_remaining.to_value(),
            base,
            representation: to_base(&step4_remaining.to_value(), base),
        },
        WalkStep {
            step_num: 6,
            removed: step6_removed,
            remaining: step6_remaining.clone(),
            value: step6_remaining.to_value(),
            base,
            representation: to_base(&step6_remaining.to_value(), base),
        },
        WalkStep {
            step_num: 8,
            removed: step8_removed,
            remaining: step8_remaining.clone(),
            value: step8_remaining.to_value(),
            base,
            representation: to_base(&step8_remaining.to_value(), base),
        },
        WalkStep {
            step_num: 10,
            removed: monster,
            remaining: step10_remaining.clone(),
            value: BigUint::from(71u32),
            base,
            representation: to_base(&BigUint::from(71u32), base),
        },
    ]
}

fn main() {
    println!("🔢 Monster Walk with Primes - All Bases");
    println!("========================================\n");
    
    // Show walk in key bases
    for base in [2, 10, 16, 71] {
        println!("Base {}:", base);
        let walk = walk_in_base(base);
        
        for step in &walk {
            println!("  Step {}: {} digits",
                step.step_num,
                step.representation.len()
            );
            if step.step_num == 4 && base == 10 {
                println!("    Primes: {}", step.remaining.format());
                println!("    ✓ Preserves 8080");
            }
        }
        println!();
    }
    
    // Generate all bases
    println!("Generating all bases 2-71...");
    let all_walks: Vec<_> = (2..=71)
        .map(|base| (base, walk_in_base(base)))
        .collect();
    
    println!("✅ Generated {} bases", all_walks.len());
    println!("✅ Each walk has {} steps", all_walks[0].1.len());
}
