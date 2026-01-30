// Optimized Prolog Searcher in Rust with Parquet Index
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize)]
struct SearchResult {
    file: String,
    line: usize,
    content: String,
    shard: u8,
    score: f64,
}

// Load parquet index
fn load_parquet_index(path: &Path) -> Result<DataFrame, PolarsError> {
    ParquetReader::new(std::fs::File::open(path)?)
        .finish()
}

// Search using parquet index
fn search_with_index(index: &DataFrame, term: &str) -> Vec<SearchResult> {
    let mut results = Vec::new();
    
    // Filter by term (would use actual parquet filtering)
    if let Ok(files) = index.column("file") {
        if let Ok(contents) = index.column("content") {
            for i in 0..files.len() {
                if let (Some(file), Some(content)) = (
                    files.get(i).ok().and_then(|v| v.get_str()),
                    contents.get(i).ok().and_then(|v| v.get_str())
                ) {
                    if content.contains(term) {
                        let shard = (file.bytes().sum::<u8>()) % 71;
                        results.push(SearchResult {
                            file: file.to_string(),
                            line: i,
                            content: content.to_string(),
                            shard,
                            score: 1.0,
                        });
                    }
                }
            }
        }
    }
    
    results
}

// Execute via zkerdfa URL
fn execute_via_zkerdfa_url(url: &str) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
    // Parse zkerdfa URL: zkerdfa://search?term=perf&index=parquet
    let parts: Vec<&str> = url.split('?').collect();
    if parts.len() != 2 {
        return Err("Invalid zkerdfa URL".into());
    }
    
    let params: std::collections::HashMap<_, _> = parts[1]
        .split('&')
        .filter_map(|p| {
            let kv: Vec<_> = p.split('=').collect();
            if kv.len() == 2 {
                Some((kv[0], kv[1]))
            } else {
                None
            }
        })
        .collect();
    
    let term = params.get("term").ok_or("Missing term")?;
    let index_path = params.get("index").ok_or("Missing index")?;
    
    // Load index
    let index = load_parquet_index(Path::new(index_path))?;
    
    // Search
    Ok(search_with_index(&index, term))
}

// Prolog integration
fn execute_prolog_query(query: &str, results: &[SearchResult]) -> String {
    let mut prolog = String::from("% Prolog facts from search\n");
    
    for (i, result) in results.iter().enumerate() {
        prolog.push_str(&format!(
            "search_result({}, '{}', {}, '{}', {}).\n",
            i, result.file, result.line, 
            result.content.replace('\'', "\\'"), result.shard
        ));
    }
    
    prolog.push_str("\n% Query\n");
    prolog.push_str(&format!("?- {}.\n", query));
    
    prolog
}

fn main() {
    println!("🔍 Optimized Prolog Searcher (Rust + Parquet)");
    println!("{}", "=".repeat(70));
    println!();
    
    // Example: zkerdfa URL
    let url = "zkerdfa://search?term=perf&index=vectors_layer_27.parquet";
    
    println!("📡 Executing zkerdfa URL:");
    println!("  {}", url);
    println!();
    
    match execute_via_zkerdfa_url(url) {
        Ok(results) => {
            println!("✓ Found {} results", results.len());
            println!();
            
            // Show first 10
            for result in results.iter().take(10) {
                println!("📄 {}:{}", result.file, result.line);
                println!("   Shard: {}", result.shard);
                println!("   {}", &result.content[..result.content.len().min(80)]);
                println!();
            }
            
            // Generate Prolog
            let prolog = execute_prolog_query(
                "search_result(_, File, _, Content, Shard), Shard < 10",
                &results
            );
            
            std::fs::write("search_results.pl", prolog).unwrap();
            println!("💾 Saved: search_results.pl");
            
            // Save JSON
            let json = serde_json::to_string_pretty(&results).unwrap();
            std::fs::write("search_results.json", json).unwrap();
            println!("💾 Saved: search_results.json");
        }
        Err(e) => {
            eprintln!("❌ Error: {}", e);
        }
    }
    
    println!();
    println!("∞ Parquet Index. Prolog Integration. zkerdfa URL. Optimized. ∞");
}
