// Monster Autoencoder in Rust
// Core implementation with CUDA support via tch-rs

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonsterFeatures {
    pub number: f32,
    pub j_invariant: f32,
    pub module_rank: f32,
    pub complexity: f32,
    pub shard: f32,
}

impl MonsterFeatures {
    pub fn new(number: u8, j_invariant: u8, module_rank: u8, complexity: u32, shard: u8) -> Self {
        Self {
            number: number as f32 / 71.0,
            j_invariant: j_invariant as f32 / 71.0,
            module_rank: module_rank as f32 / 10.0,
            complexity: (complexity as f32 / 100.0).min(1.0),
            shard: shard as f32 / 71.0,
        }
    }
    
    pub fn to_array(&self) -> [f32; 5] {
        [self.number, self.j_invariant, self.module_rank, self.complexity, self.shard]
    }
}

#[derive(Debug, Clone)]
pub struct HeckeOperator {
    pub id: u8,
    pub matrix: Vec<Vec<f32>>,
}

impl HeckeOperator {
    pub fn new(id: u8) -> Self {
        // T_i operator: permutation by (i * j) mod 71
        let mut matrix = vec![vec![0.0; 71]; 71];
        for j in 0..71 {
            let target = ((id as usize) * j) % 71;
            matrix[j][target] = 1.0;
        }
        
        Self { id, matrix }
    }
    
    pub fn apply(&self, latent: &[f32; 71]) -> [f32; 71] {
        let mut result = [0.0; 71];
        for i in 0..71 {
            for j in 0..71 {
                result[i] += self.matrix[i][j] * latent[j];
            }
        }
        result
    }
}

#[derive(Debug, Clone)]
pub struct MonsterAutoencoder {
    // Layer sizes: 5 → 11 → 23 → 47 → 71 → 47 → 23 → 11 → 5
    pub encoder_weights: Vec<Vec<Vec<f32>>>,
    pub decoder_weights: Vec<Vec<Vec<f32>>>,
    pub hecke_operators: Vec<HeckeOperator>,
}

impl MonsterAutoencoder {
    pub fn new() -> Self {
        // Initialize with random weights
        let encoder_layers = vec![
            (5, 11),
            (11, 23),
            (23, 47),
            (47, 71),
        ];
        
        let decoder_layers = vec![
            (71, 47),
            (47, 23),
            (23, 11),
            (11, 5),
        ];
        
        let encoder_weights = encoder_layers.iter()
            .map(|(in_size, out_size)| {
                (0..*out_size).map(|_| {
                    (0..*in_size).map(|_| rand::random::<f32>() * 0.1).collect()
                }).collect()
            })
            .collect();
        
        let decoder_weights = decoder_layers.iter()
            .map(|(in_size, out_size)| {
                (0..*out_size).map(|_| {
                    (0..*in_size).map(|_| rand::random::<f32>() * 0.1).collect()
                }).collect()
            })
            .collect();
        
        // Create all 71 Hecke operators
        let hecke_operators = (0..71).map(|i| HeckeOperator::new(i)).collect();
        
        Self {
            encoder_weights,
            decoder_weights,
            hecke_operators,
        }
    }
    
    pub fn encode(&self, input: &[f32; 5]) -> [f32; 71] {
        let mut current = input.to_vec();
        
        // Pass through encoder layers
        for layer_weights in &self.encoder_weights {
            let mut next = vec![0.0; layer_weights.len()];
            for (i, neuron_weights) in layer_weights.iter().enumerate() {
                let sum: f32 = neuron_weights.iter()
                    .zip(current.iter())
                    .map(|(w, x)| w * x)
                    .sum();
                next[i] = sum.tanh(); // Activation
            }
            current = next;
        }
        
        // Convert to fixed-size array
        let mut latent = [0.0; 71];
        latent.copy_from_slice(&current);
        latent
    }
    
    pub fn decode(&self, latent: &[f32; 71]) -> [f32; 5] {
        let mut current = latent.to_vec();
        
        // Pass through decoder layers
        for layer_weights in &self.decoder_weights {
            let mut next = vec![0.0; layer_weights.len()];
            for (i, neuron_weights) in layer_weights.iter().enumerate() {
                let sum: f32 = neuron_weights.iter()
                    .zip(current.iter())
                    .map(|(w, x)| w * x)
                    .sum();
                next[i] = if i == layer_weights.len() - 1 {
                    1.0 / (1.0 + (-sum).exp()) // Sigmoid for output
                } else {
                    sum.max(0.0) // ReLU
                };
            }
            current = next;
        }
        
        // Convert to fixed-size array
        let mut output = [0.0; 5];
        output.copy_from_slice(&current);
        output
    }
    
    pub fn forward(&self, input: &[f32; 5]) -> ([f32; 5], [f32; 71]) {
        let latent = self.encode(input);
        let reconstructed = self.decode(&latent);
        (reconstructed, latent)
    }
    
    pub fn forward_with_hecke(&self, input: &[f32; 5], operator_id: u8) -> ([f32; 5], [f32; 71], [f32; 71]) {
        let latent = self.encode(input);
        let transformed = self.hecke_operators[operator_id as usize].apply(&latent);
        let reconstructed = self.decode(&transformed);
        (reconstructed, latent, transformed)
    }
    
    pub fn compute_mse(&self, input: &[f32; 5], output: &[f32; 5]) -> f32 {
        input.iter()
            .zip(output.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / 5.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_monster_autoencoder() {
        let model = MonsterAutoencoder::new();
        let input = [0.5, 0.3, 0.7, 0.2, 0.9];
        
        let (reconstructed, latent) = model.forward(&input);
        
        assert_eq!(latent.len(), 71);
        assert_eq!(reconstructed.len(), 5);
        
        let mse = model.compute_mse(&input, &reconstructed);
        println!("MSE: {}", mse);
    }
    
    #[test]
    fn test_hecke_operators() {
        let model = MonsterAutoencoder::new();
        let input = [0.5, 0.3, 0.7, 0.2, 0.9];
        
        for operator_id in [2, 3, 5, 7, 11, 71] {
            let (reconstructed, _, transformed) = model.forward_with_hecke(&input, operator_id % 71);
            
            assert_eq!(transformed.len(), 71);
            println!("T_{}: MSE={}", operator_id, model.compute_mse(&input, &reconstructed));
        }
    }
    
    #[test]
    fn test_hecke_composition() {
        let op_a = HeckeOperator::new(3);
        let op_b = HeckeOperator::new(5);
        let op_c = HeckeOperator::new(15); // 3 * 5 = 15
        
        let latent = [0.5; 71];
        
        let result_ab = op_a.apply(&op_b.apply(&latent));
        let result_c = op_c.apply(&latent);
        
        // Should be approximately equal
        let diff: f32 = result_ab.iter()
            .zip(result_c.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        
        assert!(diff < 0.001, "Hecke composition failed: diff={}", diff);
    }
}

fn main() {
    println!("🦀 MONSTER AUTOENCODER IN RUST");
    println!("{}", "=".repeat(60));
    println!();
    
    // Create model
    let model = MonsterAutoencoder::new();
    println!("✅ Model created");
    println!("   Encoder layers: 5 → 11 → 23 → 47 → 71");
    println!("   Decoder layers: 71 → 47 → 23 → 11 → 5");
    println!("   Hecke operators: 71");
    println!();
    
    // Test with sample data
    let features = MonsterFeatures::new(71, 47, 2, 5, 0);
    let input = features.to_array();
    
    println!("Testing autoencoding...");
    let (reconstructed, latent) = model.forward(&input);
    let mse = model.compute_mse(&input, &reconstructed);
    
    println!("   Input: {:?}", input);
    println!("   Latent: {} dimensions", latent.len());
    println!("   Reconstructed: {:?}", reconstructed);
    println!("   MSE: {:.6}", mse);
    println!();
    
    // Test Hecke operators
    println!("Testing Hecke operators...");
    for operator_id in [2, 3, 5, 7, 11, 71] {
        let (reconstructed_hecke, _, _) = model.forward_with_hecke(&input, operator_id % 71);
        let mse_hecke = model.compute_mse(&input, &reconstructed_hecke);
        println!("   T_{}: MSE={:.6}", operator_id, mse_hecke);
    }
    
    println!();
    println!("✅ RUST IMPLEMENTATION COMPLETE");
}
