// Convert GAP JSON output to Parquet with zkprologml
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Deserialize, Serialize)]
struct PrimeAttributes {
    prime: u64,
    genus: i32,
    is_supersingular: bool,
    is_monster_prime: bool,
    mod_4: u8,
    mod_8: u8,
    mod_12: u8,
    mod_24: u8,
    mod_71: u8,
    zk71_shard: u8,
    security_zone: u8,
    nu_cusps: u32,
    nu_elliptic_2: u8,
    nu_elliptic_3: u8,
}

#[derive(Debug, Deserialize)]
struct PrimeData {
    primes: Vec<PrimeAttributes>,
}

fn main() {
    println!("🔢 Converting GAP Prime Attributes to Parquet");
    println!("{}", "=".repeat(70));
    println!();
    
    // Read JSON from GAP
    println!("📖 Reading prime_attributes.json...");
    let json_data = fs::read_to_string("prime_attributes.json")
        .expect("Failed to read prime_attributes.json");
    
    let data: PrimeData = serde_json::from_str(&json_data)
        .expect("Failed to parse JSON");
    
    println!("✓ Loaded {} primes", data.primes.len());
    println!();
    
    // Create DataFrame
    let df = DataFrame::new(vec![
        Series::new("prime", data.primes.iter().map(|p| p.prime).collect::<Vec<_>>()),
        Series::new("genus", data.primes.iter().map(|p| p.genus).collect::<Vec<_>>()),
        Series::new("is_supersingular", data.primes.iter().map(|p| p.is_supersingular).collect::<Vec<_>>()),
        Series::new("is_monster_prime", data.primes.iter().map(|p| p.is_monster_prime).collect::<Vec<_>>()),
        Series::new("mod_4", data.primes.iter().map(|p| p.mod_4).collect::<Vec<_>>()),
        Series::new("mod_8", data.primes.iter().map(|p| p.mod_8).collect::<Vec<_>>()),
        Series::new("mod_12", data.primes.iter().map(|p| p.mod_12).collect::<Vec<_>>()),
        Series::new("mod_24", data.primes.iter().map(|p| p.mod_24).collect::<Vec<_>>()),
        Series::new("mod_71", data.primes.iter().map(|p| p.mod_71).collect::<Vec<_>>()),
        Series::new("zk71_shard", data.primes.iter().map(|p| p.zk71_shard).collect::<Vec<_>>()),
        Series::new("security_zone", data.primes.iter().map(|p| p.security_zone).collect::<Vec<_>>()),
        Series::new("nu_cusps", data.primes.iter().map(|p| p.nu_cusps).collect::<Vec<_>>()),
        Series::new("nu_elliptic_2", data.primes.iter().map(|p| p.nu_elliptic_2).collect::<Vec<_>>()),
        Series::new("nu_elliptic_3", data.primes.iter().map(|p| p.nu_elliptic_3).collect::<Vec<_>>()),
    ]).unwrap();
    
    println!("📋 Sample data:");
    println!("{}", df.head(Some(10)));
    println!();
    
    // Statistics
    let genus_0 = data.primes.iter().filter(|p| p.genus == 0).count();
    let supersingular = data.primes.iter().filter(|p| p.is_supersingular).count();
    let monster = data.primes.iter().filter(|p| p.is_monster_prime).count();
    
    println!("📊 Statistics:");
    println!("  Genus 0 primes: {} (GOOD)", genus_0);
    println!("  Supersingular primes: {}", supersingular);
    println!("  Monster primes: {}", monster);
    println!();
    
    // Write to Parquet
    let mut file = fs::File::create("prime_attributes.parquet").unwrap();
    ParquetWriter::new(&mut file).finish(&mut df.clone()).unwrap();
    
    println!("✓ Saved: prime_attributes.parquet");
    
    // Generate zkprologml facts
    println!();
    println!("📝 Generating zkprologml facts...");
    
    let mut prolog = String::new();
    prolog.push_str("% Prime attributes from GAP\n\n");
    
    for p in data.primes.iter().take(20) {
        prolog.push_str(&format!(
            "prime_attr({}, genus({}), supersingular({}), monster({}), shard({}), zone({})).\n",
            p.prime, p.genus, p.is_supersingular, p.is_monster_prime, p.zk71_shard, p.security_zone
        ));
    }
    
    fs::write("prime_attributes.pl", prolog).unwrap();
    println!("✓ Saved: prime_attributes.pl");
    
    println!();
    println!("∞ GAP → Parquet → zkprologml. All Primes. All Attributes. ∞");
}
