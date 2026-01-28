//! Rust GCD implementation for bisimulation testing

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

fn main() {
    println!("Running Rust GCD...");
    
    let mut results = Vec::new();
    
    for i in 0..1000 {
        let a = (2u64.pow(i as u32)) % 71;
        let b = (3u64.pow(i as u32)) % 71;
        let g = gcd(a, b);
        results.push(g);
    }
    
    println!("Completed {} GCD computations", results.len());
    println!("Sample results: {:?}", &results[..10]);
}
