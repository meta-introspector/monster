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
    println!("COMPLETE MONSTER WALK - ALL GROUPS");
    println!("{}", "=".repeat(70));
    
    let full_str = monster_order.to_string();
    println!("\nFull Monster: {}", monster_order);
    println!("Total digits: {}\n", full_str.len());
    
    let mut position = 0;
    let mut group_num = 1;
    let mut all_groups = Vec::new();
    
    while position < full_str.len() {
        println!("{}", "=".repeat(70));
        println!("GROUP {}: Position {}", group_num, position);
        println!("{}", "=".repeat(70));
        
        let remaining = &full_str[position..];
        println!("Remaining: {}...", if remaining.len() > 30 { 
            &remaining[0..30] 
        } else { 
            remaining 
        });
        
        // Find maximum preservable digits
        let mut max_digits = 0;
        let mut best_removal = vec![];
        let mut best_num_remove = 0;
        
        for num_remove in 1..=10 {
            for num_digits in (max_digits + 1)..=10 {
                if num_digits > remaining.len() {
                    break;
                }
                
                let target = &remaining[0..num_digits];
                
                if let Some(indices) = find_combination(&primes, target, num_digits, num_remove) {
                    max_digits = num_digits;
                    best_removal = indices;
                    best_num_remove = num_remove;
                }
            }
        }
        
        if max_digits == 0 {
            println!("✗ Cannot preserve any digits - END OF WALK\n");
            break;
        }
        
        let preserved = &remaining[0..max_digits];
        println!("\n✓ Maximum: {} digits preserved", max_digits);
        println!("  Sequence: {}", preserved);
        println!("  Remove {} factors:", best_num_remove);
        
        let removed_emojis = ["🌓", "🔺", "⭐", "🎰", "🎸", "🌙", "🎯", "🎭", "🧬", "📅", "🎃", "🔮", "🎲", "⏰", "🌊"];
        
        for &idx in &best_removal {
            let (p, e) = primes[idx];
            println!("    {} {}^{}", removed_emojis[idx], p, e);
        }
        
        // Verify cannot extend
        if max_digits < remaining.len() {
            let next_target = &remaining[0..max_digits + 1];
            let mut can_extend = false;
            for num_remove in 1..=15 {
                if find_combination(&primes, next_target, max_digits + 1, num_remove).is_some() {
                    can_extend = true;
                    break;
                }
            }
            
            if !can_extend {
                println!("  ✓ VERIFIED: Cannot extend to {} digits", max_digits + 1);
            }
        }
        
        all_groups.push((group_num, position, preserved.to_string(), best_num_remove, best_removal.clone()));
        
        position += max_digits;
        group_num += 1;
        println!();
        
        if group_num > 20 {
            println!("Stopping after 20 groups...");
            break;
        }
    }
    
    // Summary
    println!("{}", "=".repeat(70));
    println!("SUMMARY: {} GROUPS FOUND", all_groups.len());
    println!("{}", "=".repeat(70));
    
    for (num, pos, seq, removals, _) in &all_groups {
        println!("Group {}: pos {} - \"{}\" ({} digits, {} removals)", 
                 num, pos, seq, seq.len(), removals);
    }
    
    println!("\n{}", "=".repeat(70));
    println!("The Monster Walk: {}", 
             all_groups.iter().map(|(_, _, s, _, _)| s.as_str()).collect::<Vec<_>>().join(" → "));
    println!("{}", "=".repeat(70));
}

fn find_combination(primes: &[(u32, u32)], target: &str, num_digits: usize, num_remove: usize) -> Option<Vec<usize>> {
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
            return Some(indices);
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
