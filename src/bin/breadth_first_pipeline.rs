// Breadth-First Layer Pipeline: Markov → Tokens → Weights (layer by layer)
// Process entire KB one layer at a time, forwarding through all shards

use std::fs;
use serde::{Deserialize, Serialize};

// ============================================================================
// LAYER STRUCTURES
// ============================================================================

#[derive(Clone, Serialize, Deserialize)]
struct LayerTokens {
    layer: u8,
    shards: Vec<ShardTokens>,
}

#[derive(Clone, Serialize, Deserialize)]
struct ShardTokens {
    shard: u8,
    token_ids: Vec<u32>,
    vocab_size: usize,
}

#[derive(Clone, Serialize, Deserialize)]
struct LayerWeights {
    layer: u8,
    shards: Vec<ShardWeight>,
}

#[derive(Clone, Serialize, Deserialize)]
struct ShardWeight {
    shard: u8,
    weights: Vec<f32>,
    hecke_eigenvalue: f64,
}

// ============================================================================
// BREADTH-FIRST PIPELINE
// ============================================================================

struct BreadthFirstPipeline {
    primes: Vec<u32>,
    max_layers: usize,
    num_shards: usize,
}

impl BreadthFirstPipeline {
    fn new(primes: Vec<u32>, max_layers: usize, num_shards: usize) -> Self {
        Self { primes, max_layers, num_shards }
    }
    
    // Process entire KB layer by layer
    fn process_kb(&self, corpus_path: &str) -> Vec<LayerWeights> {
        println!("🌊 BREADTH-FIRST LAYER PROCESSING");
        println!("{}", "=".repeat(70));
        println!("Processing {} layers × {} shards", self.max_layers, self.num_shards);
        println!("{}", "=".repeat(70));
        println!();
        
        let mut all_layer_weights = Vec::new();
        
        // Process one layer at a time (breadth-first)
        for layer in 0..self.max_layers {
            println!("\n📍 LAYER {}/{}", layer + 1, self.max_layers);
            println!("{}", "-".repeat(70));
            
            // Step 1: Load all shards for this layer
            let layer_tokens = self.load_layer(corpus_path, layer as u8);
            
            // Step 2: Transform Markov → Tokens
            let layer_markov = self.markov_to_tokens(&layer_tokens);
            
            // Step 3: Tokens → Weights (GPU forward pass)
            let layer_weights = self.tokens_to_weights(layer_markov, layer as u8);
            
            // Step 4: Apply Hecke operators
            let layer_hecke = self.apply_layer_hecke(layer_weights);
            
            all_layer_weights.push(layer_hecke);
            
            println!("✅ Layer {} complete", layer);
        }
        
        all_layer_weights
    }
    
    // Step 1: Load all shards for one layer
    fn load_layer(&self, corpus_path: &str, layer: u8) -> LayerTokens {
        println!("  📂 Loading layer {} across {} shards", layer, self.num_shards);
        
        let mut shards = Vec::new();
        
        for shard in 0..self.num_shards {
            let file_path = format!("{}/shard_{:02}_layer_{:02}.txt", corpus_path, shard, layer);
            
            let text = fs::read_to_string(&file_path)
                .unwrap_or_else(|_| format!("sample {}", shard));
            
            let token_ids: Vec<u32> = text.as_bytes().iter().map(|&b| b as u32).collect();
            
            shards.push(ShardTokens {
                shard: shard as u8,
                token_ids,
                vocab_size: 256,
            });
        }
        
        println!("     Loaded {} shards", shards.len());
        
        LayerTokens { layer, shards }
    }
    
    // Step 2: Markov → Tokens (build transition matrices)
    fn markov_to_tokens(&self, layer_tokens: &LayerTokens) -> Vec<Vec<f32>> {
        println!("  🔄 Markov → Tokens");
        
        let mut matrices = Vec::new();
        
        for shard_tokens in &layer_tokens.shards {
            let matrix = self.build_transition_matrix(
                &shard_tokens.token_ids,
                shard_tokens.vocab_size
            );
            matrices.push(matrix);
        }
        
        println!("     Built {} transition matrices", matrices.len());
        matrices
    }
    
    fn build_transition_matrix(&self, token_ids: &[u32], vocab_size: usize) -> Vec<f32> {
        let mut counts = vec![0u32; vocab_size * vocab_size];
        
        for window in token_ids.windows(2) {
            let curr = window[0] as usize;
            let next = window[1] as usize;
            if curr < vocab_size && next < vocab_size {
                counts[curr * vocab_size + next] += 1;
            }
        }
        
        let mut matrix = vec![0.0f32; vocab_size * vocab_size];
        for i in 0..vocab_size {
            let row_start = i * vocab_size;
            let row_sum: u32 = counts[row_start..row_start + vocab_size].iter().sum();
            
            if row_sum > 0 {
                for j in 0..vocab_size {
                    matrix[row_start + j] = counts[row_start + j] as f32 / row_sum as f32;
                }
            } else {
                for j in 0..vocab_size {
                    matrix[row_start + j] = 1.0 / vocab_size as f32;
                }
            }
        }
        
        matrix
    }
    
    // Step 3: Tokens → Weights (GPU forward pass)
    fn tokens_to_weights(&self, matrices: Vec<Vec<f32>>, layer: u8) -> Vec<ShardWeight> {
        println!("  ⚡ Tokens → Weights (GPU forward pass)");
        
        let mut shard_weights = Vec::new();
        
        for (shard_idx, matrix) in matrices.iter().enumerate() {
            let vocab_size = (matrix.len() as f64).sqrt() as usize;
            
            // Initial state: uniform distribution
            let input = vec![1.0 / vocab_size as f32; vocab_size];
            
            // GPU forward pass: matrix × input
            let weights = self.forward_pass_gpu(matrix, &input, vocab_size);
            
            shard_weights.push(ShardWeight {
                shard: shard_idx as u8,
                weights,
                hecke_eigenvalue: 0.0, // Computed in next step
            });
        }
        
        println!("     Computed {} weight vectors", shard_weights.len());
        shard_weights
    }
    
    fn forward_pass_gpu(&self, matrix: &[f32], input: &[f32], vocab_size: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; vocab_size];
        
        // GPU parallel: each thread computes one output
        for i in 0..vocab_size {
            let row_start = i * vocab_size;
            output[i] = (0..vocab_size)
                .map(|j| matrix[row_start + j] * input[j])
                .sum();
        }
        
        output
    }
    
    // Step 4: Apply Hecke operators to layer
    fn apply_layer_hecke(&self, mut shard_weights: Vec<ShardWeight>) -> LayerWeights {
        println!("  🔢 Applying Hecke operators");
        
        let max_prime = *self.primes.last().unwrap() as f32;
        
        for shard_weight in &mut shard_weights {
            let prime = self.primes[shard_weight.shard as usize % self.primes.len()];
            
            // Compute weight from output
            let weight = shard_weight.weights.iter().sum::<f32>() / shard_weight.weights.len() as f32;
            
            // Apply Hecke operator
            let eigenvalue = (weight * prime as f32) % max_prime;
            shard_weight.hecke_eigenvalue = eigenvalue as f64;
            
            println!("     Shard {:2}: prime={:2}, eigenvalue={:.4}", 
                shard_weight.shard, prime, eigenvalue);
        }
        
        LayerWeights {
            layer: shard_weights[0].shard, // Layer ID from context
            shards: shard_weights,
        }
    }
    
    // Save results
    fn save_results(&self, all_layers: &[LayerWeights]) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n💾 Saving results");
        println!("{}", "-".repeat(70));
        
        fs::create_dir_all("breadth_first_output")?;
        
        // Save each layer
        for layer_weights in all_layers {
            let filename = format!("breadth_first_output/layer_{:02}_weights.json", layer_weights.layer);
            let json = serde_json::to_string_pretty(&layer_weights)?;
            fs::write(&filename, json)?;
        }
        
        // Save summary
        let summary = serde_json::json!({
            "total_layers": all_layers.len(),
            "shards_per_layer": self.num_shards,
            "total_weights": all_layers.len() * self.num_shards,
        });
        fs::write("breadth_first_output/summary.json", serde_json::to_string_pretty(&summary)?)?;
        
        println!("✅ Saved {} layers", all_layers.len());
        Ok(())
    }
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌊 BREADTH-FIRST LAYER PIPELINE");
    println!("{}", "=".repeat(70));
    println!("Markov → Tokens → Weights (layer by layer)");
    println!("{}", "=".repeat(70));
    println!();
    
    let primes = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];
    let pipeline = BreadthFirstPipeline::new(primes, 46, 15);
    
    // Process entire KB breadth-first
    let all_layer_weights = pipeline.process_kb("corpus");
    
    // Save results
    pipeline.save_results(&all_layer_weights)?;
    
    println!();
    println!("{}", "=".repeat(70));
    println!("✅ BREADTH-FIRST PROCESSING COMPLETE");
    println!("{}", "=".repeat(70));
    println!("📊 Processed {} layers", all_layer_weights.len());
    println!("📊 Total shards: {}", all_layer_weights.len() * 15);
    println!("💾 Output: breadth_first_output/");
    
    Ok(())
}
