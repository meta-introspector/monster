use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

const MONSTER_PRIMES: [u32; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];

#[derive(Debug, Serialize)]
struct CodeToMonsterMap {
    total_codes: usize,
    mapped_codes: Vec<MappedCode>,
    prime_distribution: HashMap<u32, usize>,
    monster_clusters: Vec<MonsterCluster>,
}

#[derive(Debug, Serialize)]
struct MappedCode {
    code_id: String,
    name: String,
    dimension: Option<usize>,
    prime_signature: Vec<u32>,
    monster_connection: String,
}

#[derive(Debug, Serialize)]
struct MonsterCluster {
    primes: Vec<u32>,
    codes: Vec<String>,
    size: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎪 Error Correction Zoo → Monster Group Mapping");
    println!("================================================\n");
    
    let eczoo_path = Path::new("/home/mdupont/experiments/monster/examples/eczoo_data/codes");
    
    println!("Scanning error correction codes...\n");
    
    let mut mapping = CodeToMonsterMap {
        total_codes: 0,
        mapped_codes: Vec::new(),
        prime_distribution: HashMap::new(),
        monster_clusters: Vec::new(),
    };
    
    // Walk all YAML files
    for entry in walkdir::WalkDir::new(eczoo_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|s| s == "yml").unwrap_or(false))
    {
        mapping.total_codes += 1;
        
        if let Ok(content) = fs::read_to_string(entry.path()) {
            if let Some(mapped) = map_code_to_monster(&content) {
                // Update prime distribution
                for &prime in &mapped.prime_signature {
                    *mapping.prime_distribution.entry(prime).or_insert(0) += 1;
                }
                
                mapping.mapped_codes.push(mapped);
            }
        }
        
        if mapping.total_codes % 100 == 0 {
            println!("  Processed {} codes...", mapping.total_codes);
        }
    }
    
    println!("\n✓ Scanned {} codes", mapping.total_codes);
    println!("  Mapped to Monster: {}", mapping.mapped_codes.len());
    
    // Build Monster clusters
    mapping.monster_clusters = build_clusters(&mapping.mapped_codes);
    
    // Print results
    println!("\n📊 Prime Distribution:");
    let mut primes: Vec<_> = mapping.prime_distribution.iter().collect();
    primes.sort_by_key(|(p, _)| **p);
    
    for (prime, count) in primes {
        let is_monster = MONSTER_PRIMES.contains(prime);
        let marker = if is_monster { "🎪" } else { "  " };
        println!("  {} Prime {}: {} codes", marker, prime, count);
    }
    
    println!("\n🎪 Monster Clusters:");
    for (i, cluster) in mapping.monster_clusters.iter().take(10).enumerate() {
        println!("  Cluster {}: primes={:?}, codes={}", 
                 i+1, cluster.primes, cluster.size);
    }
    
    // Find Leech lattice
    println!("\n🔮 Special Codes:");
    for code in &mapping.mapped_codes {
        if code.code_id.contains("leech") {
            println!("  ✓ Leech lattice: {:?} → {}", 
                     code.prime_signature, code.monster_connection);
        }
        if code.code_id.contains("golay") {
            println!("  ✓ Golay code: {:?} → {}", 
                     code.prime_signature, code.monster_connection);
        }
    }
    
    // Save mapping
    fs::write(
        "CODE_MONSTER_MAP.json",
        serde_json::to_string_pretty(&mapping)?
    )?;
    
    println!("\n✓ Mapping complete!");
    println!("  Results: CODE_MONSTER_MAP.json");
    
    Ok(())
}

fn map_code_to_monster(yaml_content: &str) -> Option<MappedCode> {
    // Parse YAML (simple extraction)
    let code_id = extract_field(yaml_content, "code_id:")?;
    let name = extract_field(yaml_content, "name:").unwrap_or_default();
    
    // Extract dimension
    let dimension = extract_dimension(yaml_content);
    
    // Map to Monster primes based on properties
    let prime_signature = compute_prime_signature(yaml_content, dimension);
    
    if prime_signature.is_empty() {
        return None;
    }
    
    // Determine Monster connection
    let connection = determine_connection(&code_id, &prime_signature);
    
    Some(MappedCode {
        code_id,
        name,
        dimension,
        prime_signature,
        monster_connection: connection,
    })
}

fn extract_field(content: &str, field: &str) -> Option<String> {
    content.lines()
        .find(|line| line.trim_start().starts_with(field))
        .and_then(|line| line.split(':').nth(1))
        .map(|s| s.trim().trim_matches('\'').trim_matches('"').to_string())
}

fn extract_dimension(content: &str) -> Option<usize> {
    // Look for dimension indicators
    if content.contains("24") { return Some(24); }
    if content.contains("12") { return Some(12); }
    if content.contains("8") { return Some(8); }
    None
}

fn compute_prime_signature(content: &str, dimension: Option<usize>) -> Vec<u32> {
    let mut primes = Vec::new();
    
    // Dimension-based mapping
    if let Some(dim) = dimension {
        match dim {
            24 => primes.extend(&[2, 3]),  // Leech lattice
            12 => primes.push(3),           // Golay
            8 => primes.push(2),            // E8
            _ => {}
        }
    }
    
    // Content-based mapping
    let lower = content.to_lowercase();
    
    if lower.contains("leech") { primes.extend(&[2, 3, 5, 7, 11]); }
    if lower.contains("golay") { primes.extend(&[2, 3, 11]); }
    if lower.contains("conway") { primes.extend(&[2, 3, 5, 7]); }
    if lower.contains("monster") { primes.extend(&MONSTER_PRIMES); }
    if lower.contains("moonshine") { primes.extend(&[2, 3, 5, 7, 11, 13]); }
    
    primes.sort();
    primes.dedup();
    primes
}

fn determine_connection(code_id: &str, primes: &[u32]) -> String {
    if code_id.contains("leech") {
        "Direct: Leech lattice → Conway group → Monster".to_string()
    } else if code_id.contains("golay") {
        "Direct: Golay code → Mathieu group → Monster".to_string()
    } else if primes.len() >= 5 {
        "Strong: Multiple Monster primes".to_string()
    } else if primes.len() >= 2 {
        "Moderate: Some Monster primes".to_string()
    } else {
        "Weak: Few Monster primes".to_string()
    }
}

fn build_clusters(codes: &[MappedCode]) -> Vec<MonsterCluster> {
    let mut clusters: HashMap<Vec<u32>, Vec<String>> = HashMap::new();
    
    for code in codes {
        clusters.entry(code.prime_signature.clone())
            .or_insert_with(Vec::new)
            .push(code.code_id.clone());
    }
    
    let mut result: Vec<_> = clusters.into_iter()
        .map(|(primes, codes)| MonsterCluster {
            size: codes.len(),
            primes,
            codes,
        })
        .collect();
    
    result.sort_by_key(|c| std::cmp::Reverse(c.size));
    result
}
