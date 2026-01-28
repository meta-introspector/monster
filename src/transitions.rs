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
    println!("MONSTER WALK: DIGIT TRANSITION ANALYSIS");
    println!("{}", "=".repeat(70));
    println!("\nFull Monster: {}", monster_order);
    
    let full_str = monster_order.to_string();
    
    // Analyze transitions between digit groups
    println!("\n{}", "=".repeat(70));
    println!("TRANSITION 1: Remove leading '8' to expose '0801742...'");
    println!("{}", "=".repeat(70));
    
    analyze_digit_removal(&primes, &full_str, 1, "0");
    
    println!("\n{}", "=".repeat(70));
    println!("TRANSITION 2: Remove leading '80' to expose '801742...'");
    println!("{}", "=".repeat(70));
    
    analyze_digit_removal(&primes, &full_str, 2, "80");
    
    println!("\n{}", "=".repeat(70));
    println!("TRANSITION 3: Remove '8080' to expose next digits");
    println!("{}", "=".repeat(70));
    
    println!("After '8080', the sequence is: {}", &full_str[4..]);
    println!("Searching for what we CAN expose...\n");
    
    // Find what actually starts after removing factors
    for num_remove in 1..=6 {
        if let Some((indices, reduced)) = find_best_after_8080(&primes, num_remove) {
            let removed_primes: Vec<_> = indices.iter().map(|&i| primes[i].0).collect();
            let result_str = reduced.to_string();
            
            println!("✓ Remove {} factors: {:?}", num_remove, removed_primes);
            println!("  Result starts with: {}", &result_str[0..10.min(result_str.len())]);
            
            // Check if it matches the target sequence after 8080
            let target_after = &full_str[4..];
            let matching = count_matching_prefix(&result_str, target_after);
            if matching >= 4 {
                println!("  ★ MATCHES {} digits of target sequence!", matching);
                for &i in &indices {
                    let (p, e) = primes[i];
                    println!("    {}^{} = {}", p, e, BigUint::from(p).pow(e));
                }
                break;
            }
        }
    }
    
    println!("\n{}", "=".repeat(70));
    println!("TRANSITION 4: After '80801742', expose next digits");
    println!("{}", "=".repeat(70));
    
    println!("After '80801742', the sequence is: {}", &full_str[8..]);
    println!("Searching for what we CAN expose...\n");
    
    for num_remove in 1..=6 {
        if let Some((indices, reduced)) = find_best_after_80801742(&primes, num_remove) {
            let removed_primes: Vec<_> = indices.iter().map(|&i| primes[i].0).collect();
            let result_str = reduced.to_string();
            
            println!("✓ Remove {} factors: {:?}", num_remove, removed_primes);
            println!("  Result starts with: {}", &result_str[0..10.min(result_str.len())]);
            
            let target_after = &full_str[8..];
            let matching = count_matching_prefix(&result_str, target_after);
            if matching >= 3 {
                println!("  ★ MATCHES {} digits of target sequence!", matching);
                for &i in &indices {
                    let (p, e) = primes[i];
                    println!("    {}^{} = {}", p, e, BigUint::from(p).pow(e));
                }
                break;
            }
        }
    }
}

fn analyze_digit_removal(primes: &[(u32, u32)], full_str: &str, skip: usize, target_start: &str) {
    println!("Target: Start with '{}'", target_start);
    println!("Searching for factor combinations...\n");
    
    let mut found_any = false;
    
    for num_remove in 1..=10 {
        if let Some((indices, reduced)) = find_starts_with(primes, target_start, num_remove) {
            let removed_primes: Vec<_> = indices.iter().map(|&i| primes[i].0).collect();
            
            println!("✓ Remove {} factors: {:?}", num_remove, removed_primes);
            println!("  Factors:");
            for &i in &indices {
                let (p, e) = primes[i];
                println!("    {}^{} = {}", p, e, BigUint::from(p).pow(e));
            }
            println!("  Result: {}", reduced);
            
            let result_str = reduced.to_string();
            let matching = count_matching_prefix(&result_str, target_start);
            println!("  Matching digits: {}", matching);
            
            found_any = true;
            break;
        }
    }
    
    if !found_any {
        println!("✗ No combination found to start with '{}'", target_start);
    }
}

fn find_starts_with(primes: &[(u32, u32)], target_start: &str, num_remove: usize) -> Option<(Vec<usize>, BigUint)> {
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
        if result_str.starts_with(target_start) {
            return Some((indices.clone(), reduced_order));
        }
        
        if !next_combination(&mut indices, n) {
            break;
        }
    }
    
    None
}

fn count_matching_prefix(s: &str, target: &str) -> usize {
    s.chars().zip(target.chars()).take_while(|(a, b)| a == b).count()
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

fn find_best_after_8080(primes: &[(u32, u32)], num_remove: usize) -> Option<(Vec<usize>, BigUint)> {
    let n = primes.len();
    let mut indices = vec![0; num_remove];
    let target = "1742";
    
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
        if result_str.starts_with(target) {
            return Some((indices.clone(), reduced_order));
        }
        
        if !next_combination(&mut indices, n) {
            break;
        }
    }
    None
}

fn find_best_after_80801742(primes: &[(u32, u32)], num_remove: usize) -> Option<(Vec<usize>, BigUint)> {
    let n = primes.len();
    let mut indices = vec![0; num_remove];
    let target = "479";
    
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
        if result_str.starts_with(target) {
            return Some((indices.clone(), reduced_order));
        }
        
        if !next_combination(&mut indices, n) {
            break;
        }
    }
    None
}
