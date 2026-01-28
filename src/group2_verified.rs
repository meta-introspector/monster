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
    println!("MONSTER WALK GROUP 2: After 8080 - VERIFIED MAXIMUM");
    println!("{}", "=".repeat(70));
    
    let full_str = monster_order.to_string();
    let after_8080 = &full_str[4..];
    
    println!("\nTarget sequence: {}", after_8080);
    println!("\nSearching for MAXIMUM digit preservation...\n");
    
    let mut max_digits_found = 0;
    let mut best_removal = vec![];
    let mut best_result = BigUint::one();
    
    for num_remove in 1..=10 {
        println!("Trying {} factor removals...", num_remove);
        
        for start_digits in (max_digits_found + 1)..=10 {
            let target = &after_8080[0..start_digits.min(after_8080.len())];
            
            if let Some((indices, reduced)) = find_combination(&primes, target, start_digits, num_remove) {
                max_digits_found = start_digits;
                best_removal = indices.clone();
                best_result = reduced.clone();
                println!("  ✓ Found {} digits preserved!", start_digits);
            }
        }
    }
    
    println!("\n{}", "=".repeat(70));
    println!("MAXIMUM RESULT FOR GROUP 2");
    println!("{}", "=".repeat(70));
    println!("\nMaximum digits preserved: {}", max_digits_found);
    println!("Preserved sequence: {}", &after_8080[0..max_digits_found]);
    println!("\nFactors removed ({} total):", best_removal.len());
    for &i in &best_removal {
        let (p, e) = primes[i];
        println!("  {}^{} = {}", p, e, BigUint::from(p).pow(e));
    }
    println!("\nResult: {}", best_result);
    
    println!("\n{}", "=".repeat(70));
    println!("VERIFICATION: Cannot preserve {} digits", max_digits_found + 1);
    println!("{}", "=".repeat(70));
    let next_target = &after_8080[0..(max_digits_found + 1).min(after_8080.len())];
    println!("Would need: {}", next_target);
    
    let mut found_next = false;
    for num_remove in 1..=15 {
        if find_combination(&primes, next_target, max_digits_found + 1, num_remove).is_some() {
            found_next = true;
            break;
        }
    }
    
    if found_next {
        println!("✗ ERROR: Found {} digits!", max_digits_found + 1);
    } else {
        println!("✓ CONFIRMED: No combination preserves {} digits", max_digits_found + 1);
        println!("   This is the MAXIMUM for Group 2");
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
