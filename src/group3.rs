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
    println!("MONSTER WALK GROUP 3: After 80801742");
    println!("{}", "=".repeat(70));
    println!("\nFull Monster order: {}", monster_order);
    
    let full_str = monster_order.to_string();
    let after_80801742 = &full_str[8..]; // Skip "80801742"
    
    println!("Digits after '80801742': {}", after_80801742);
    println!("Length: {} digits\n", after_80801742.len());
    
    for start_digits in 1..=10 {
        let target = &after_80801742[0..start_digits.min(after_80801742.len())];
        println!("Searching for {} leading digits: {}", start_digits, target);
        
        let mut best_found = false;
        for num_remove in 1..=10 {
            if let Some((indices, reduced)) = find_combination(&primes, target, start_digits, num_remove) {
                let removed_primes: Vec<_> = indices.iter().map(|&i| primes[i].0).collect();
                println!("  ✓ {} removals preserve {} digits: {:?}", num_remove, start_digits, removed_primes);
                
                if num_remove <= 3 {
                    println!("    Result: {}", reduced);
                    for &i in &indices {
                        let (p, e) = primes[i];
                        println!("    Remove: {}^{} = {}", p, e, BigUint::from(p).pow(e));
                    }
                }
                best_found = true;
                break;
            }
        }
        
        if !best_found {
            println!("  ✗ No combination found");
            break;
        }
    }
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
