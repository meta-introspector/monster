use num_bigint::BigUint;
use num_traits::One;

fn main() {
    let primes = vec![
        (2u32, 46u32), (3, 20), (5, 9), (7, 6), (11, 2), (13, 3),
        (17, 1), (19, 1), (23, 1), (29, 1), (31, 1), (41, 1),
        (47, 1), (59, 1), (71, 1),
    ];

    let mut monster_order = BigUint::one();
    for (prime, exponent) in &primes {
        monster_order *= BigUint::from(*prime).pow(*exponent);
    }

    println!("{}", "=".repeat(70));
    println!("MONSTER WALK: DIFFERENT BASE REPRESENTATIONS");
    println!("{}", "=".repeat(70));
    
    // Base 10 (decimal)
    println!("\n>>> BASE 10 (Decimal)");
    analyze_base(&primes, &monster_order, 10);
    
    // Base 2 (binary)
    println!("\n>>> BASE 2 (Binary)");
    analyze_base(&primes, &monster_order, 2);
    
    // Base 16 (hexadecimal)
    println!("\n>>> BASE 16 (Hexadecimal)");
    analyze_base(&primes, &monster_order, 16);
    
    // Base 8 (octal)
    println!("\n>>> BASE 8 (Octal)");
    analyze_base(&primes, &monster_order, 8);
    
    // Base 3
    println!("\n>>> BASE 3 (Ternary)");
    analyze_base(&primes, &monster_order, 3);
    
    // Base 7
    println!("\n>>> BASE 7");
    analyze_base(&primes, &monster_order, 7);
}

fn analyze_base(primes: &[(u32, u32)], monster_order: &BigUint, base: u32) {
    let base_repr = to_base(monster_order, base);
    println!("Monster in base {}: {}", base, 
             if base_repr.len() > 50 { 
                 format!("{}... ({} digits)", &base_repr[0..50], base_repr.len())
             } else { 
                 base_repr.clone() 
             });
    
    // Try to preserve leading digits in this base
    for num_digits in 1..=6 {
        let target = &base_repr[0..num_digits.min(base_repr.len())];
        
        for num_remove in 1..=8 {
            if let Some((indices, _)) = find_preserves_base(primes, target, base, num_remove) {
                let removed_primes: Vec<_> = indices.iter().map(|&i| primes[i].0).collect();
                println!("  ✓ Base {}: {} digits preserved by removing {} factors: {:?}", 
                         base, num_digits, num_remove, removed_primes);
                break;
            }
        }
    }
}

fn to_base(n: &BigUint, base: u32) -> String {
    if n.is_zero() {
        return "0".to_string();
    }
    
    let mut result = String::new();
    let mut num = n.clone();
    let base_big = BigUint::from(base);
    
    while !num.is_zero() {
        let digit = (&num % &base_big).to_u32_digits();
        let d = if digit.is_empty() { 0 } else { digit[0] };
        
        result.push(if d < 10 {
            (b'0' + d as u8) as char
        } else {
            (b'A' + (d - 10) as u8) as char
        });
        
        num /= &base_big;
    }
    
    result.chars().rev().collect()
}

fn find_preserves_base(primes: &[(u32, u32)], target: &str, base: u32, num_remove: usize) -> Option<(Vec<usize>, BigUint)> {
    let n = primes.len();
    let mut indices = vec![0; num_remove];
    
    for i in 0..num_remove {
        indices[i] = i;
    }
    
    loop {
        let mut reduced_order = BigUint::one();
        for i in 0..n {
            if !indices.contains(&i) {
                let (p, e) = primes[i];
                reduced_order *= BigUint::from(p).pow(e);
            }
        }
        
        let base_repr = to_base(&reduced_order, base);
        if base_repr.starts_with(target) {
            return Some((indices.clone(), reduced_order));
        }
        
        if !next_combination(&mut indices, n) {
            break;
        }
    }
    
    None
}

fn next_combination(indices: &mut [usize], n: usize) -> bool {
    let k = indices.len();
    for i in (0..k).rev() {
        if indices[i] < n - k + i {
            indices[i] += 1;
            for j in (i + 1)..k {
                indices[j] = indices[j - 1] + 1;
            }
            return true;
        }
    }
    false
}
