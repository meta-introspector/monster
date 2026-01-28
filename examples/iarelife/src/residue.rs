use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

const MONSTER_PRIMES: [u32; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];

#[derive(Debug, Serialize)]
struct ResidueAnalysis {
    total_codes: usize,
    mapped_codes: usize,
    unmapped_codes: Vec<UnmappedCode>,
    residue_patterns: Vec<ResiduePattern>,
    missing_primes: Vec<u32>,
    found_primes: Vec<u32>,
}

#[derive(Debug, Serialize)]
struct UnmappedCode {
    code_id: String,
    name: String,
    reason: String,
    properties: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ResiduePattern {
    pattern: String,
    count: usize,
    examples: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔍 Monster Residue Analysis");
    println!("===========================\n");
    
    let eczoo_path = Path::new("/home/mdupont/experiments/monster/examples/eczoo_data/codes");
    
    let mut residue = ResidueAnalysis {
        total_codes: 0,
        mapped_codes: 0,
        unmapped_codes: Vec::new(),
        residue_patterns: Vec::new(),
        missing_primes: Vec::new(),
        found_primes: Vec::new(),
    };
    
    let mut all_primes_found = HashSet::new();
    
    println!("Analyzing all codes...\n");
    
    for entry in walkdir::WalkDir::new(eczoo_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|s| s == "yml").unwrap_or(false))
    {
        residue.total_codes += 1;
        
        if let Ok(content) = fs::read_to_string(entry.path()) {
            let primes = extract_primes(&content);
            
            if !primes.is_empty() {
                residue.mapped_codes += 1;
                all_primes_found.extend(primes);
            } else {
                if let Some(unmapped) = analyze_unmapped(&content) {
                    residue.unmapped_codes.push(unmapped);
                }
            }
        }
    }
    
    // Calculate found and missing primes
    residue.found_primes = all_primes_found.iter()
        .filter(|p| MONSTER_PRIMES.contains(p))
        .copied()
        .collect();
    residue.found_primes.sort();
    
    residue.missing_primes = MONSTER_PRIMES.iter()
        .filter(|p| !all_primes_found.contains(p))
        .copied()
        .collect();
    
    println!("✓ Analyzed {} codes", residue.total_codes);
    println!("  Mapped: {}", residue.mapped_codes);
    println!("  Residue: {}", residue.unmapped_codes.len());
    
    // Find patterns in residue
    residue.residue_patterns = find_residue_patterns(&residue.unmapped_codes);
    
    println!("\n📊 Residue Patterns:");
    for pattern in &residue.residue_patterns {
        println!("  {}: {} codes", pattern.pattern, pattern.count);
        for example in pattern.examples.iter().take(3) {
            println!("    - {}", example);
        }
    }
    
    println!("\n🎪 Found Monster Primes:");
    for prime in &residue.found_primes {
        println!("  ✓ Prime {}", prime);
    }
    
    println!("\n❌ Missing Monster Primes:");
    for prime in &residue.missing_primes {
        println!("  ✗ Prime {}", prime);
    }
    
    // Calculate percentages
    let mapped_pct = (residue.mapped_codes as f64 / residue.total_codes as f64) * 100.0;
    let residue_pct = (residue.unmapped_codes.len() as f64 / residue.total_codes as f64) * 100.0;
    let prime_coverage = (residue.found_primes.len() as f64 / MONSTER_PRIMES.len() as f64) * 100.0;
    
    println!("\n📈 Statistics:");
    println!("  Total codes: {}", residue.total_codes);
    println!("  Mapped to Monster: {} ({:.1}%)", residue.mapped_codes, mapped_pct);
    println!("  Residue: {} ({:.1}%)", residue.unmapped_codes.len(), residue_pct);
    println!("  Prime coverage: {}/{} ({:.1}%)", 
             residue.found_primes.len(), MONSTER_PRIMES.len(), prime_coverage);
    
    // Save
    fs::write("RESIDUE_ANALYSIS.json", serde_json::to_string_pretty(&residue)?)?;
    
    let report = generate_report(&residue);
    fs::write("RESIDUE_REPORT.md", report)?;
    
    println!("\n✓ Analysis complete!");
    println!("  JSON: RESIDUE_ANALYSIS.json");
    println!("  Report: RESIDUE_REPORT.md");
    
    Ok(())
}

fn extract_primes(content: &str) -> Vec<u32> {
    let mut primes = Vec::new();
    let lower = content.to_lowercase();
    
    // Dimension-based
    if content.contains("24") { primes.extend(&[2, 3]); }
    if content.contains("12") { primes.push(3); }
    if content.contains("8") { primes.push(2); }
    
    // Keyword-based
    if lower.contains("leech") { primes.extend(&[2, 3, 5, 7, 11]); }
    if lower.contains("golay") { primes.extend(&[2, 3, 11]); }
    if lower.contains("conway") { primes.extend(&[2, 3, 5, 7]); }
    if lower.contains("monster") { primes.extend(&MONSTER_PRIMES); }
    
    primes.sort();
    primes.dedup();
    primes
}

fn analyze_unmapped(content: &str) -> Option<UnmappedCode> {
    let code_id = extract_field(content, "code_id:")?;
    let name = extract_field(content, "name:").unwrap_or_default();
    let reason = determine_unmapped_reason(content);
    let properties = extract_properties(content);
    
    Some(UnmappedCode { code_id, name, reason, properties })
}

fn determine_unmapped_reason(content: &str) -> String {
    let lower = content.to_lowercase();
    
    if lower.contains("quantum") && !lower.contains("classical") {
        "Quantum-only code".to_string()
    } else if lower.contains("ldpc") {
        "LDPC code".to_string()
    } else if lower.contains("turbo") {
        "Turbo code".to_string()
    } else if lower.contains("polar") {
        "Polar code".to_string()
    } else {
        "No Monster connection".to_string()
    }
}

fn extract_properties(content: &str) -> Vec<String> {
    let mut props = Vec::new();
    if content.contains("quantum") { props.push("quantum".to_string()); }
    if content.contains("classical") { props.push("classical".to_string()); }
    if content.contains("ldpc") { props.push("ldpc".to_string()); }
    props
}

fn extract_field(content: &str, field: &str) -> Option<String> {
    content.lines()
        .find(|line| line.trim_start().starts_with(field))
        .and_then(|line| line.split(':').nth(1))
        .map(|s| s.trim().trim_matches('\'').trim_matches('"').to_string())
}

fn find_residue_patterns(unmapped: &[UnmappedCode]) -> Vec<ResiduePattern> {
    let mut patterns: HashMap<String, Vec<String>> = HashMap::new();
    
    for code in unmapped {
        patterns.entry(code.reason.clone())
            .or_insert_with(Vec::new)
            .push(code.code_id.clone());
    }
    
    let mut result: Vec<_> = patterns.into_iter()
        .map(|(pattern, examples)| ResiduePattern {
            count: examples.len(),
            pattern,
            examples: examples.into_iter().take(5).collect(),
        })
        .collect();
    
    result.sort_by_key(|p| std::cmp::Reverse(p.count));
    result
}

fn generate_report(residue: &ResidueAnalysis) -> String {
    let mut report = String::from("# Monster Group Residue Analysis\n\n");
    
    let mapped_pct = (residue.mapped_codes as f64 / residue.total_codes as f64) * 100.0;
    let residue_pct = (residue.unmapped_codes.len() as f64 / residue.total_codes as f64) * 100.0;
    
    report.push_str("## Summary\n\n");
    report.push_str(&format!("- **Total codes**: {}\n", residue.total_codes));
    report.push_str(&format!("- **Mapped to Monster**: {} ({:.1}%)\n", residue.mapped_codes, mapped_pct));
    report.push_str(&format!("- **Residue**: {} ({:.1}%)\n\n", residue.unmapped_codes.len(), residue_pct));
    
    report.push_str("## Monster Prime Coverage\n\n");
    report.push_str(&format!("Found {}/{} Monster primes:\n\n", 
                            residue.found_primes.len(), MONSTER_PRIMES.len()));
    
    for prime in &residue.found_primes {
        report.push_str(&format!("- ✓ **{}**\n", prime));
    }
    
    report.push_str("\nMissing primes:\n\n");
    for prime in &residue.missing_primes {
        report.push_str(&format!("- ✗ **{}**\n", prime));
    }
    
    report.push_str("\n## Residue Patterns\n\n");
    for pattern in &residue.residue_patterns {
        report.push_str(&format!("### {} ({} codes)\n\n", pattern.pattern, pattern.count));
        for example in &pattern.examples {
            report.push_str(&format!("- `{}`\n", example));
        }
        report.push_str("\n");
    }
    
    report
}
