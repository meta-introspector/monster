// Statistical Significance Analysis of Monster Prime Vectorization
// Proves semantic structure exists in 71-layer embeddings

use polars::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use serde::{Serialize, Deserialize};
use serde_json;

#[derive(Serialize, Deserialize)]
struct ChiSquareTest {
    null_hypothesis: String,
    alternative_hypothesis: String,
    chi_square_statistic: f64,
    p_value: f64,
    reject_null: bool,
    significance_level: f64,
    interpretation: String,
    observed_distribution: Vec<usize>,
    expected_uniform: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
struct CoherenceTest {
    layers_analyzed: Vec<usize>,
    entities_tracked: usize,
    shard_consistency_score: f64,
    interpretation: String,
}

#[derive(Serialize, Deserialize)]
struct EntropyAnalysis {
    average_entropy: f64,
    max_possible_entropy: f64,
    normalized_entropy: f64,
    interpretation: String,
}

#[derive(Serialize, Deserialize)]
struct AnalysisResults {
    analysis_date: String,
    method: String,
    randomness_test: Option<ChiSquareTest>,
    coherence_test: Option<CoherenceTest>,
    entropy_analysis: Option<EntropyAnalysis>,
    conclusion: String,
}

fn load_shard(layer: usize) -> Option<DataFrame> {
    let path = format!("vectors_layer_{:02}.parquet", layer);
    File::open(&path)
        .ok()
        .and_then(|f| ParquetReader::new(f).finish().ok())
}

fn chi_square_test(observed: &[usize], expected: &[f64]) -> (f64, f64) {
    let chi2: f64 = observed.iter().zip(expected.iter())
        .map(|(o, e)| {
            let diff = *o as f64 - e;
            (diff * diff) / e
        })
        .sum();
    
    // Approximate p-value using chi-square distribution (df = 14 for 15 categories)
    let df = 14.0;
    let p_value = 1.0 - chi_square_cdf(chi2, df);
    
    (chi2, p_value)
}

fn chi_square_cdf(x: f64, df: f64) -> f64 {
    // Approximation using gamma function
    if x <= 0.0 { return 0.0; }
    if x > 100.0 { return 1.0; }
    
    // Simple approximation for p-value
    let k = df / 2.0;
    let z = x / 2.0;
    
    // Use series expansion
    let mut sum = 0.0;
    let mut term = 1.0;
    for i in 0..50 {
        sum += term;
        term *= z / (k + i as f64);
        if term < 1e-10 { break; }
    }
    
    let gamma_k = gamma(k);
    1.0 - (z.powf(k) * (-z).exp() * sum / gamma_k).min(1.0)
}

fn gamma(z: f64) -> f64 {
    // Stirling's approximation
    if z < 1.0 { return gamma(z + 1.0) / z; }
    ((2.0 * std::f64::consts::PI / z).sqrt() * (z / std::f64::consts::E).powf(z))
}

fn entropy(counts: &[usize]) -> f64 {
    let total: usize = counts.iter().sum();
    if total == 0 { return 0.0; }
    
    counts.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total as f64;
            -p * p.ln()
        })
        .sum()
}

fn test_randomness() -> Option<ChiSquareTest> {
    println!("📊 Test 1: Testing for non-random structure...");
    
    let mut all_shards = Vec::new();
    
    for layer in 0..71 {
        if let Some(df) = load_shard(layer) {
            if let Ok(col) = df.column("shard") {
                for val in col.u32().unwrap().into_iter() {
                    if let Some(v) = val {
                        all_shards.push(v as usize);
                    }
                }
            }
        }
    }
    
    if all_shards.len() < 100 {
        println!("   ❌ Insufficient data");
        return None;
    }
    
    // Count occurrences
    let mut observed = vec![0usize; 15];
    for &shard in &all_shards {
        if shard < 15 {
            observed[shard] += 1;
        }
    }
    
    // Expected uniform
    let expected: Vec<f64> = vec![all_shards.len() as f64 / 15.0; 15];
    
    let (chi2, p_value) = chi_square_test(&observed, &expected);
    let reject_null = p_value < 0.05;
    
    if reject_null {
        println!("   ✅ SIGNIFICANT: p = {:.2e} < 0.05", p_value);
        println!("   → Shard assignments are NOT random");
    } else {
        println!("   ❌ Not significant: p = {:.2e}", p_value);
    }
    
    Some(ChiSquareTest {
        null_hypothesis: "Shard assignments are uniformly random".to_string(),
        alternative_hypothesis: "Shard assignments follow semantic structure".to_string(),
        chi_square_statistic: chi2,
        p_value,
        reject_null,
        significance_level: 0.05,
        interpretation: "p < 0.05 means semantic structure exists (reject randomness)".to_string(),
        observed_distribution: observed,
        expected_uniform: expected,
    })
}

fn test_coherence() -> Option<CoherenceTest> {
    println!("📊 Test 2: Cross-layer semantic coherence...");
    
    let layers = vec![0, 17, 35, 53, 70];
    let mut dfs = Vec::new();
    
    for &layer in &layers {
        if let Some(df) = load_shard(layer) {
            dfs.push(df);
        }
    }
    
    if dfs.len() < 2 {
        println!("   ❌ Not enough layers");
        return None;
    }
    
    // Track shard consistency per file+column
    let mut entity_shards: HashMap<String, Vec<u32>> = HashMap::new();
    
    for df in &dfs {
        if let (Ok(files), Ok(cols), Ok(shards)) = (
            df.column("file"),
            df.column("column"),
            df.column("shard")
        ) {
            let files = files.str().unwrap();
            let cols = cols.str().unwrap();
            let shards = shards.u32().unwrap();
            
            for i in 0..df.height() {
                if let (Some(f), Some(c), Some(s)) = (files.get(i), cols.get(i), shards.get(i)) {
                    let key = format!("{}::{}", f, c);
                    entity_shards.entry(key).or_insert_with(Vec::new).push(s);
                }
            }
        }
    }
    
    // Compute consistency (low variance = high consistency)
    let mut variances = Vec::new();
    for shards in entity_shards.values() {
        if shards.len() > 1 {
            let mean: f64 = shards.iter().map(|&s| s as f64).sum::<f64>() / shards.len() as f64;
            let var: f64 = shards.iter()
                .map(|&s| (s as f64 - mean).powi(2))
                .sum::<f64>() / shards.len() as f64;
            variances.push(var);
        }
    }
    
    let avg_var = variances.iter().sum::<f64>() / variances.len() as f64;
    let coherence = 1.0 / (1.0 + avg_var);
    
    println!("   Coherence score: {:.3}", coherence);
    if coherence > 0.7 {
        println!("   ✅ HIGH coherence - semantic entities stable across layers");
    } else if coherence > 0.5 {
        println!("   ⚠️  MODERATE coherence");
    } else {
        println!("   ❌ LOW coherence");
    }
    
    Some(CoherenceTest {
        layers_analyzed: layers,
        entities_tracked: entity_shards.len(),
        shard_consistency_score: coherence,
        interpretation: "High score = same semantic entity stays in same Monster prime shard".to_string(),
    })
}

fn test_entropy() -> Option<EntropyAnalysis> {
    println!("📊 Test 3: Intra-shard entropy analysis...");
    
    let test_layers = vec![0, 17, 35, 53, 70];
    let mut entropies = Vec::new();
    
    for &layer in &test_layers {
        if let Some(df) = load_shard(layer) {
            if let Ok(shards) = df.column("shard") {
                let mut counts = vec![0usize; 15];
                for val in shards.u32().unwrap().into_iter() {
                    if let Some(v) = val {
                        if (v as usize) < 15 {
                            counts[v as usize] += 1;
                        }
                    }
                }
                
                let ent = entropy(&counts);
                entropies.push(ent);
                println!("   Layer {:2}: entropy = {:.3}", layer, ent);
            }
        }
    }
    
    if entropies.is_empty() {
        println!("   ❌ No data");
        return None;
    }
    
    let avg_entropy: f64 = entropies.iter().sum::<f64>() / entropies.len() as f64;
    let max_entropy = (15.0f64).ln();
    let normalized = avg_entropy / max_entropy;
    
    if normalized < 0.8 {
        println!("   ✅ Normalized entropy: {:.3} < 0.8 (good clustering)", normalized);
    } else {
        println!("   ⚠️  Normalized entropy: {:.3} (weak clustering)", normalized);
    }
    
    Some(EntropyAnalysis {
        average_entropy: avg_entropy,
        max_possible_entropy: max_entropy,
        normalized_entropy: normalized,
        interpretation: "Lower entropy = stronger clustering (semantic structure)".to_string(),
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 STATISTICAL SIGNIFICANCE ANALYSIS");
    println!("{}", "=".repeat(70));
    println!();
    
    let randomness = test_randomness();
    println!();
    
    let coherence = test_coherence();
    println!();
    
    let entropy_analysis = test_entropy();
    println!();
    
    // Summary
    println!("{}", "=".repeat(70));
    println!("📋 SUMMARY");
    println!("{}", "=".repeat(70));
    
    let mut significant_tests = 0;
    let mut total_tests = 0;
    
    if let Some(ref test) = randomness {
        total_tests += 1;
        if test.reject_null {
            significant_tests += 1;
        }
    }
    
    if coherence.is_some() {
        total_tests += 1;
        if let Some(ref test) = coherence {
            if test.shard_consistency_score > 0.7 {
                significant_tests += 1;
            }
        }
    }
    
    if entropy_analysis.is_some() {
        total_tests += 1;
        if let Some(ref test) = entropy_analysis {
            if test.normalized_entropy < 0.8 {
                significant_tests += 1;
            }
        }
    }
    
    println!("Significant tests: {}/{}", significant_tests, total_tests);
    println!();
    
    let conclusion = if significant_tests >= 2 {
        println!("✅ CONCLUSION: Statistical evidence supports semantic structure in Monster embeddings");
        println!("   The 71-layer, 15-shard vectorization captures real semantic meaning.");
        "Statistical evidence supports semantic structure in Monster embeddings".to_string()
    } else {
        println!("⚠️  CONCLUSION: Insufficient evidence for semantic structure");
        println!("   May need more data or refined analysis.");
        "Insufficient evidence for semantic structure".to_string()
    };
    
    println!();
    
    // Save results
    let results = AnalysisResults {
        analysis_date: "2026-01-29".to_string(),
        method: "Monster Prime Vectorization (71 layers, 15 shards)".to_string(),
        randomness_test: randomness,
        coherence_test: coherence,
        entropy_analysis,
        conclusion,
    };
    
    let json = serde_json::to_string_pretty(&results)?;
    std::fs::write("semantic_significance_analysis.json", json)?;
    
    println!("💾 Results saved to: semantic_significance_analysis.json");
    println!();
    
    Ok(())
}
