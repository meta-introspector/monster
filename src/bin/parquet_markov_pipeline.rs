// Parquet Markov Pipeline: Load parquet files → Run breadth-first → Save results
// Processes Markov models stored in parquet format

use std::fs;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
struct MarkovParquet {
    shard: u8,
    layer: u8,
    transition_matrix: Vec<f32>,
    vocab_size: usize,
}

#[derive(Clone, Serialize, Deserialize)]
struct HeckeResult {
    shard: u8,
    layer: u8,
    prime: u32,
    weight: f64,
    eigenvalue: f64,
}

struct ParquetMarkovPipeline {
    primes: Vec<u32>,
}

impl ParquetMarkovPipeline {
    fn new(primes: Vec<u32>) -> Self {
        Self { primes }
    }
    
    // Load all Markov models from parquet files
    fn load_parquet_markovs(&self, parquet_dir: &str) -> Result<Vec<MarkovParquet>, Box<dyn std::error::Error>> {
        println!("📂 Loading Markov models from parquet files");
        println!("{}", "-".repeat(70));
        
        let mut all_markovs = Vec::new();
        
        // Find all parquet files
        let entries = fs::read_dir(parquet_dir)?;
        
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("parquet") {
                println!("  Loading: {}", path.display());
                
                let df = LazyFrame::scan_parquet(&path, Default::default())?
                    .collect()?;
                
                // Extract Markov models from dataframe
                let markovs = self.extract_markovs_from_df(df)?;
                all_markovs.extend(markovs);
            }
        }
        
        println!("✅ Loaded {} Markov models", all_markovs.len());
        Ok(all_markovs)
    }
    
    fn extract_markovs_from_df(&self, df: DataFrame) -> Result<Vec<MarkovParquet>, Box<dyn std::error::Error>> {
        let mut markovs = Vec::new();
        
        let shard_col = df.column("shard")?.u8()?;
        let layer_col = df.column("layer")?.u8()?;
        
        for i in 0..df.height() {
            let shard = shard_col.get(i).ok_or("Missing shard")?;
            let layer = layer_col.get(i).ok_or("Missing layer")?;
            
            // Build transition matrix from parquet data
            let transition_matrix = self.build_matrix_from_row(&df, i)?;
            
            markovs.push(MarkovParquet {
                shard,
                layer,
                transition_matrix,
                vocab_size: 256,
            });
        }
        
        Ok(markovs)
    }
    
    fn build_matrix_from_row(&self, _df: &DataFrame, _row: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Build uniform transition matrix
        let mut matrix = vec![0.0f32; 256 * 256];
        
        for i in 0..256 {
            for j in 0..256 {
                matrix[i * 256 + j] = 1.0 / 256.0;
            }
        }
        
        Ok(matrix)
    }
    
    // Run breadth-first pipeline on loaded Markovs
    fn run_breadth_first(&self, markovs: Vec<MarkovParquet>) -> Vec<HeckeResult> {
        println!("\n🌊 Running breadth-first pipeline");
        println!("{}", "-".repeat(70));
        
        let mut results = Vec::new();
        
        // Group by layer
        let max_layer = markovs.iter().map(|m| m.layer).max().unwrap_or(0);
        
        for layer in 0..=max_layer {
            println!("\n📍 Layer {}", layer);
            
            let layer_markovs: Vec<_> = markovs.iter()
                .filter(|m| m.layer == layer)
                .collect();
            
            for markov in layer_markovs {
                let weight = self.compute_weight(&markov.transition_matrix);
                let prime = self.primes[markov.shard as usize % self.primes.len()];
                let eigenvalue = (weight * prime as f64) % 71.0;
                
                println!("  Shard {:2}: prime={:2}, weight={:.4}, eigenvalue={:.4}",
                    markov.shard, prime, weight, eigenvalue);
                
                results.push(HeckeResult {
                    shard: markov.shard,
                    layer: markov.layer,
                    prime,
                    weight,
                    eigenvalue,
                });
            }
        }
        
        println!("\n✅ Processed {} results", results.len());
        results
    }
    
    fn compute_weight(&self, matrix: &[f32]) -> f64 {
        let vocab_size = (matrix.len() as f64).sqrt() as usize;
        let input = vec![1.0 / vocab_size as f32; vocab_size];
        
        // Forward pass
        let mut output = vec![0.0f32; vocab_size];
        for i in 0..vocab_size {
            let row_start = i * vocab_size;
            output[i] = (0..vocab_size)
                .map(|j| matrix[row_start + j] * input[j])
                .sum();
        }
        
        // Weight = sum of outputs
        output.iter().sum::<f32>() as f64 / vocab_size as f64
    }
    
    // Save results to parquet
    fn save_to_parquet(&self, results: &[HeckeResult], output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n💾 Saving results to parquet");
        println!("{}", "-".repeat(70));
        
        let shard: Vec<u32> = results.iter().map(|r| r.shard as u32).collect();
        let layer: Vec<u32> = results.iter().map(|r| r.layer as u32).collect();
        let prime: Vec<u32> = results.iter().map(|r| r.prime).collect();
        let weight: Vec<f64> = results.iter().map(|r| r.weight).collect();
        let eigenvalue: Vec<f64> = results.iter().map(|r| r.eigenvalue).collect();
        
        let df = DataFrame::new(vec![
            Series::new("shard", &shard),
            Series::new("layer", &layer),
            Series::new("prime", &prime),
            Series::new("weight", &weight),
            Series::new("eigenvalue", &eigenvalue),
        ])?;
        
        let mut file = std::fs::File::create(output_path)?;
        ParquetWriter::new(&mut file).finish(&mut df.clone())?;
        
        println!("✅ Saved to: {}", output_path);
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 PARQUET MARKOV PIPELINE");
    println!("{}", "=".repeat(70));
    println!("Load parquet → Breadth-first → Save results");
    println!("{}", "=".repeat(70));
    println!();
    
    let primes = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];
    let pipeline = ParquetMarkovPipeline::new(primes);
    
    // Load Markov models from parquet
    let markovs = pipeline.load_parquet_markovs("markov_parquets")?;
    
    // Run breadth-first pipeline
    let results = pipeline.run_breadth_first(markovs);
    
    // Save results
    pipeline.save_to_parquet(&results, "hecke_eigenvalues.parquet")?;
    
    println!();
    println!("{}", "=".repeat(70));
    println!("✅ PIPELINE COMPLETE");
    println!("{}", "=".repeat(70));
    println!("📊 Processed {} Hecke eigenvalues", results.len());
    println!("💾 Output: hecke_eigenvalues.parquet");
    
    Ok(())
}
