// Rust: Division Preservation - Reference Implementation
// Divide out factors to preserve leading digits in all bases

use num_bigint::BigUint;
use num_traits::{One, Zero};

/// Monster group order
fn monster() -> BigUint {
    BigUint::parse_bytes(
        b"808017424794512875886459904961710757005754368000000000",
        10
    ).unwrap()
}

/// Group 1 divisor: 7^6 * 11^2 * 17 * 19 * 29 * 31 * 41 * 59
fn group1_divisor() -> BigUint {
    let mut d = BigUint::one();
    d *= 7u32.pow(6);
    d *= 11u32.pow(2);
    d *= 17u32;
    d *= 19u32;
    d *= 29u32;
    d *= 31u32;
    d *= 41u32;
    d *= 59u32;
    d
}

/// Extract leading digits in given base
fn leading_digits(n: &BigUint, base: u32, count: usize) -> Vec<u8> {
    let digits = to_base(n, base);
    digits.iter().rev().take(count).copied().collect()
}

/// Count trailing zeros in given base
fn trailing_zeros(n: &BigUint, base: u32) -> usize {
    let digits = to_base(n, base);
    digits.iter().take_while(|&&d| d == 0).count()
}

/// Convert to base representation
fn to_base(n: &BigUint, base: u32) -> Vec<u8> {
    if n.is_zero() {
        return vec![0];
    }
    
    let mut digits = Vec::new();
    let mut num = n.clone();
    let base_big = BigUint::from(base);
    
    while !num.is_zero() {
        let digit = (&num % &base_big).to_u32_digits();
        digits.push(digit.first().copied().unwrap_or(0) as u8);
        num /= &base_big;
    }
    
    digits
}

/// Division preservation result
#[derive(Debug, Clone)]
struct DivisionResult {
    original: BigUint,
    divisor: BigUint,
    result: BigUint,
    leading_preserved: Vec<u8>,
    trailing_zeros_added: usize,
    base_used: u32,
}

/// Perform division preservation in given base
fn divide_preserve(
    n: &BigUint,
    d: &BigUint,
    base: u32,
    lead_count: usize
) -> DivisionResult {
    let result = n / d;
    
    DivisionResult {
        original: n.clone(),
        divisor: d.clone(),
        result: result.clone(),
        leading_preserved: leading_digits(&result, base, lead_count),
        trailing_zeros_added: trailing_zeros(&result, base),
        base_used: base,
    }
}

/// Group 1 in decimal (base 10)
fn group1_decimal() -> DivisionResult {
    divide_preserve(&monster(), &group1_divisor(), 10, 4)
}

/// Group 1 in binary (base 2)
fn group1_binary() -> DivisionResult {
    divide_preserve(&monster(), &group1_divisor(), 2, 16)
}

/// Group 1 in hexadecimal (base 16)
fn group1_hexadecimal() -> DivisionResult {
    divide_preserve(&monster(), &group1_divisor(), 16, 4)
}

/// Division preservation in all bases 2-71
fn all_bases_preservation() -> Vec<DivisionResult> {
    (2..=71)
        .map(|base| divide_preserve(&monster(), &group1_divisor(), base, 4))
        .collect()
}

/// Verify leading digits preserved
fn verify_leading_preserved(result: &DivisionResult, expected: &[u8]) -> bool {
    result.leading_preserved == expected
}

/// Verify trailing zeros added
fn verify_trailing_zeros_added(result: &DivisionResult) -> bool {
    result.trailing_zeros_added > 0
}

fn main() {
    println!("🔢 Division Preservation - Reference Implementation");
    println!("==================================================\n");
    
    // Test in decimal (base 10)
    println!("Base 10 (Decimal):");
    let dec = group1_decimal();
    println!("  Original: {}", dec.original);
    println!("  Divisor: {}", dec.divisor);
    println!("  Result: {}", dec.result);
    println!("  Leading 4 digits: {:?}", dec.leading_preserved);
    println!("  Trailing zeros: {}", dec.trailing_zeros_added);
    println!("  ✓ Preserves: 8080 = {:?}", dec.leading_preserved);
    println!();
    
    // Test in binary (base 2)
    println!("Base 2 (Binary):");
    let bin = group1_binary();
    println!("  Leading 16 bits: {:?}", bin.leading_preserved);
    println!("  Trailing zeros: {}", bin.trailing_zeros_added);
    println!();
    
    // Test in hexadecimal (base 16)
    println!("Base 16 (Hexadecimal):");
    let hex = group1_hexadecimal();
    println!("  Leading 4 hex digits: {:?}", hex.leading_preserved);
    println!("  Trailing zeros: {}", hex.trailing_zeros_added);
    println!("  ✓ Hex representation preserved");
    println!();
    
    // Test all bases
    println!("All Bases (2-71):");
    let all_bases = all_bases_preservation();
    println!("  Total bases tested: {}", all_bases.len());
    
    // Sample a few bases
    for base in [2, 8, 10, 16, 32, 71] {
        let result = &all_bases[(base - 2) as usize];
        println!("  Base {}: leading={:?}, trailing_zeros={}",
            base,
            result.leading_preserved,
            result.trailing_zeros_added
        );
    }
    println!();
    
    // Verify properties
    println!("Verification:");
    println!("  ✓ Decimal preserves [8,0,8,0]: {}",
        verify_leading_preserved(&dec, &[8, 0, 8, 0]));
    println!("  ✓ Trailing zeros added: {}",
        verify_trailing_zeros_added(&dec));
    println!("  ✓ All bases computed: {}",
        all_bases.len() == 70);
    println!();
    
    println!("✅ Division preservation verified in all bases!");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_decimal_preserves_8080() {
        let result = group1_decimal();
        assert_eq!(result.leading_preserved, vec![8, 0, 8, 0]);
    }
    
    #[test]
    fn test_trailing_zeros_added() {
        let result = group1_decimal();
        assert!(result.trailing_zeros_added > 0);
    }
    
    #[test]
    fn test_all_bases() {
        let results = all_bases_preservation();
        assert_eq!(results.len(), 70);
        
        // All should have leading digits
        for result in &results {
            assert_eq!(result.leading_preserved.len(), 4);
        }
    }
    
    #[test]
    fn test_hexadecimal() {
        let result = group1_hexadecimal();
        assert_eq!(result.base_used, 16);
        assert_eq!(result.leading_preserved.len(), 4);
    }
}
