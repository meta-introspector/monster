use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Automorphic loop detector - feeds model output back to itself
#[derive(Debug, Serialize)]
pub struct AutomorphicLoop {
    pub model: String,
    pub initial_input: String,
    pub iterations: Vec<LoopIteration>,
    pub stabilized: bool,
    pub stabilization_point: Option<usize>,
    pub cycle_length: Option<usize>,
    pub semantic_vector: Vec<f64>,
}

#[derive(Debug, Serialize, Clone)]
pub struct LoopIteration {
    pub iteration: usize,
    pub input: String,
    pub output: String,
    pub concepts: Vec<String>,
    pub similarity_to_prev: f64,
    pub tower_level: usize,
}

/// Model capacity analyzer - which models handle which tower levels
#[derive(Debug, Serialize)]
pub struct ModelCapacity {
    pub model: String,
    pub max_tower_level: usize,
    pub successful_levels: Vec<usize>,
    pub failed_levels: Vec<usize>,
    pub capacity_score: f64,
}

pub struct AutomorphicAnalyzer {
    max_iterations: usize,
    stability_threshold: f64,
}

impl AutomorphicAnalyzer {
    pub fn new(max_iterations: usize) -> Self {
        Self {
            max_iterations,
            stability_threshold: 0.95, // 95% similarity = stable
        }
    }
    
    /// Run automorphic loop test
    pub fn test_loop(&self, model: &str, initial: &str, tower_level: usize) -> Result<AutomorphicLoop> {
        let mut iterations = Vec::new();
        let mut current_input = initial.to_string();
        let mut stabilized = false;
        let mut stabilization_point = None;
        
        println!("  🔄 Testing automorphic loop at tower level {}", tower_level);
        
        for i in 0..self.max_iterations {
            // TODO: Actual model inference
            let output = self.mock_inference(model, &current_input, tower_level);
            let concepts = self.extract_concepts(&output);
            
            let similarity = if i > 0 {
                self.calculate_similarity(&iterations[i-1].output, &output)
            } else {
                0.0
            };
            
            iterations.push(LoopIteration {
                iteration: i,
                input: current_input.clone(),
                output: output.clone(),
                concepts: concepts.clone(),
                similarity_to_prev: similarity,
                tower_level,
            });
            
            // Check for stabilization
            if similarity > self.stability_threshold {
                stabilized = true;
                stabilization_point = Some(i);
                println!("    ✓ Stabilized at iteration {} (similarity: {:.2}%)", i, similarity * 100.0);
                break;
            }
            
            // Check for cycles
            if let Some(cycle) = self.detect_cycle(&iterations) {
                println!("    🔁 Cycle detected: length {}", cycle);
                return Ok(AutomorphicLoop {
                    model: model.to_string(),
                    initial_input: initial.to_string(),
                    iterations,
                    stabilized: false,
                    stabilization_point: None,
                    cycle_length: Some(cycle),
                    semantic_vector: self.compute_semantic_vector(&iterations),
                });
            }
            
            // Feed output back as input
            current_input = output;
        }
        
        if !stabilized {
            println!("    ⚠ Did not stabilize after {} iterations", self.max_iterations);
        }
        
        Ok(AutomorphicLoop {
            model: model.to_string(),
            initial_input: initial.to_string(),
            iterations,
            stabilized,
            stabilization_point,
            cycle_length: None,
            semantic_vector: self.compute_semantic_vector(&iterations),
        })
    }
    
    /// Test model capacity across tower levels
    pub fn test_capacity(&self, model: &str, max_level: usize) -> Result<ModelCapacity> {
        let mut successful = Vec::new();
        let mut failed = Vec::new();
        
        println!("  📊 Testing capacity of {} across {} tower levels", model, max_level);
        
        for level in 0..=max_level {
            let test_input = format!("Explain Monster Walk at abstraction level {}", level);
            
            // Test if model can handle this level
            let can_handle = self.test_level_capacity(model, &test_input, level)?;
            
            if can_handle {
                successful.push(level);
                println!("    ✓ Level {}: Success", level);
            } else {
                failed.push(level);
                println!("    ✗ Level {}: Failed", level);
            }
        }
        
        let max_tower_level = *successful.iter().max().unwrap_or(&0);
        let capacity_score = successful.len() as f64 / (max_level + 1) as f64;
        
        Ok(ModelCapacity {
            model: model.to_string(),
            max_tower_level,
            successful_levels: successful,
            failed_levels: failed,
            capacity_score,
        })
    }
    
    fn test_level_capacity(&self, model: &str, input: &str, level: usize) -> Result<bool> {
        // TODO: Actual inference
        // Smaller models fail at higher abstraction levels
        let model_size = if model.contains("7b") { 7 } else { 70 };
        let can_handle = level <= (model_size / 10);
        Ok(can_handle)
    }
    
    fn mock_inference(&self, _model: &str, input: &str, _level: usize) -> String {
        // TODO: Replace with actual mistral.rs inference
        format!("Response to: {}", input)
    }
    
    fn extract_concepts(&self, text: &str) -> Vec<String> {
        // Simple concept extraction
        vec!["Monster".to_string(), "group".to_string(), "periodicity".to_string()]
    }
    
    fn calculate_similarity(&self, text1: &str, text2: &str) -> f64 {
        // Simple Jaccard similarity
        let words1: std::collections::HashSet<_> = text1.split_whitespace().collect();
        let words2: std::collections::HashSet<_> = text2.split_whitespace().collect();
        
        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();
        
        if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
    }
    
    fn detect_cycle(&self, iterations: &[LoopIteration]) -> Option<usize> {
        if iterations.len() < 4 { return None; }
        
        // Check for repeating patterns
        for cycle_len in 2..=iterations.len()/2 {
            let recent = &iterations[iterations.len()-cycle_len..];
            let prev = &iterations[iterations.len()-2*cycle_len..iterations.len()-cycle_len];
            
            let matches = recent.iter().zip(prev.iter())
                .filter(|(a, b)| self.calculate_similarity(&a.output, &b.output) > 0.9)
                .count();
            
            if matches == cycle_len {
                return Some(cycle_len);
            }
        }
        
        None
    }
    
    fn compute_semantic_vector(&self, iterations: &[LoopIteration]) -> Vec<f64> {
        // Compute semantic trajectory through concept space
        let mut vector = Vec::new();
        for iter in iterations {
            vector.push(iter.concepts.len() as f64);
            vector.push(iter.similarity_to_prev);
        }
        vector
    }
}

/// Tower level analyzer - shows which models handle which levels
#[derive(Debug, Serialize)]
pub struct TowerAnalysis {
    pub total_levels: usize,
    pub model_capacities: Vec<ModelCapacity>,
    pub level_coverage: HashMap<usize, Vec<String>>, // level -> models that handle it
}

impl TowerAnalysis {
    pub fn new(capacities: Vec<ModelCapacity>, total_levels: usize) -> Self {
        let mut level_coverage: HashMap<usize, Vec<String>> = HashMap::new();
        
        for capacity in &capacities {
            for &level in &capacity.successful_levels {
                level_coverage.entry(level)
                    .or_insert_with(Vec::new)
                    .push(capacity.model.clone());
            }
        }
        
        Self {
            total_levels,
            model_capacities: capacities,
            level_coverage,
        }
    }
    
    pub fn print_summary(&self) {
        println!("\n🗼 Tower of Babel - Model Capacity Analysis");
        println!("==========================================");
        
        for level in 0..self.total_levels {
            let models = self.level_coverage.get(&level).map(|v| v.len()).unwrap_or(0);
            println!("  Level {}: {} models can handle", level, models);
            if let Some(model_list) = self.level_coverage.get(&level) {
                for model in model_list {
                    println!("    - {}", model);
                }
            }
        }
        
        println!("\n📊 Model Capacities:");
        for capacity in &self.model_capacities {
            println!("  {}: max level {}, score {:.2}%", 
                     capacity.model, 
                     capacity.max_tower_level,
                     capacity.capacity_score * 100.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_automorphic_loop() {
        let analyzer = AutomorphicAnalyzer::new(10);
        let result = analyzer.test_loop("test-model", "initial input", 0).unwrap();
        assert!(result.iterations.len() > 0);
    }
    
    #[test]
    fn test_similarity() {
        let analyzer = AutomorphicAnalyzer::new(10);
        let sim = analyzer.calculate_similarity("hello world", "hello world");
        assert_eq!(sim, 1.0);
    }
}
