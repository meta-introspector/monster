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
    println!("MONSTER WALK: PEELING THE ONION - ALL GROUPS");
    println!("{}", "=".repeat(70));
    
    let full_str = monster_order.to_string();
    println!("\nFull Monster: {}", monster_order);
    println!("Total digits: {}\n", full_str.len());
    
    let mut skip = 0;
    let mut group_num = 1;
    
    loop {
        println!("{}", "=".repeat(70));
        println!("GROUP {}: Starting at position {}", group_num, skip);
        println!("{}", "=".repeat(70));
        
        if skip >= full_str.len() {
            println!("Reached end of number!");
            break;
        }
        
        let remaining = &full_str[skip..];
        println!("Remaining digits: {}", if remaining.len() > 40 { 
            format!("{}... ({} digits)", &remaining[0..40], remaining.len())
        } else {
            remaining.to_string()
        });
        
        // Find maximum digits we can preserve
        let mut max_found = 0;
        let mut best_removal = vec![];
        
        for num_digits in 1..=10 {
            if num_digits > remaining.len() {
                break;
            }
            
            let target = &remaining[0..num_digits];
            let mut found = false;
            
            for num_remove in 1..=10 {
                if let Some((indices, _)) = find_combination(&primes, target, num_digits, num_remove) {
                    max_found = num_digits;
                    best_removal = indices;
                    found = true;
                    break;
                }
            }
            
            if !found {
                break;
            }
        }
        
        if max_found == 0 {
            println!("✗ Cannot preserve any digits - END OF WALK");
            break;
        }
        
        println!("\n✓ Maximum preserved: {} digits", max_found);
        println!("  Sequence: {}", &remaining[0..max_found]);
        println!("  Factors removed ({} total):", best_removal.len());
        for &i in &best_removal {
            let (p, e) = primes[i];
            println!("    {}^{}", p, e);
        }
        
        // Try to get one more digit
        if max_found < remaining.len() {
            let next_target = &remaining[0..max_found + 1];
            let mut can_extend = false;
            for num_remove in 1..=15 {
                if find_combination(&primes, next_target, max_found + 1, num_remove).is_some() {
                    can_extend = true;
                    break;
                }
            }
            
            if can_extend {
                println!("  ⚠ WARNING: Can extend to {} digits!", max_found + 1);
            } else {
                println!("  ✓ VERIFIED: Cannot extend to {} digits", max_found + 1);
            }
        }
        
        skip += max_found;
        group_num += 1;
        println!();
        
        if group_num > 10 {
            println!("Stopping after 10 groups...");
            break;
        }
    }
    
    println!("\n{}", "=".repeat(70));
    println!("SUMMARY: Found {} groups in the Monster Walk", group_num - 1);
    println!("{}", "=".repeat(70));
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
        
        let result_str = reduced_order.to_string();
        if result_str.len() >= num_digits && &result_str[0..num_digits] == target {
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
