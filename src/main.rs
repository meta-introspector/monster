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

    println!("Monster group order: {}", monster_order);
    println!("First 5 digits: {}", get_leading_n_digits(&monster_order, 5));
    println!("\nSearching for 5-digit preservation (80801)...\n");

    // Check if any combination preserves 5 digits
    for num_remove in 2..=15 {
        println!("Checking {} factor removals...", num_remove);
        
        if let Some((indices, reduced)) = find_combination(&primes, &get_leading_n_digits(&monster_order, 5), 5, num_remove) {
            println!("\n✓ FOUND! Removing {} factors preserves 5 digits (80801)", num_remove);
            println!("Remove:");
            for &i in &indices {
                let (p, e) = primes[i];
                let val = BigUint::from(p).pow(e);
                println!("  {}^{} = {}", p, e, val);
            }
            println!("Result: {}", reduced);
            return;
        }
    }
    
    println!("\n✗ No combination found that preserves 5 digits");
}

fn find_combination(primes: &[(u32, u32)], target: &str, num_digits: usize, num_remove: usize) -> Option<(Vec<usize>, BigUint)> {
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
        
        let new_leading = get_leading_n_digits(&reduced_order, num_digits);
        if new_leading == target {
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

fn get_leading_n_digits(n: &BigUint, num_digits: usize) -> String {
    let s = n.to_string();
    if s.len() >= num_digits {
        s[0..num_digits].to_string()
    } else {
        s
    }
}
