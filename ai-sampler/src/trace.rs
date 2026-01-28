use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use chrono::Utc;

/// Full trace of AI model interaction
#[derive(Debug, Serialize, Deserialize)]
pub struct AITrace {
    pub timestamp: String,
    pub model: String,
    pub model_size: ModelSize,
    pub input: TraceInput,
    pub output: TraceOutput,
    pub metrics: TraceMetrics,
    pub semantic_class: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ModelSize {
    Tiny,    // <1B params
    Small,   // 1-7B
    Medium,  // 7-13B
    Large,   // 13-70B
    XLarge,  // >70B
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TraceInput {
    pub modality: Modality,
    pub content: String,
    pub source_file: Option<String>,
    pub harmonic_class: Option<u32>, // Prime frequency
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Modality {
    Text,
    Vision,
    Code,
    Math,
    Proof,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TraceOutput {
    pub response: String,
    pub concepts_extracted: Vec<String>,
    pub confidence: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TraceMetrics {
    pub tokens_in: usize,
    pub tokens_out: usize,
    pub latency_ms: u64,
    pub memory_mb: f64,
}

/// Convergence analyzer across models
#[derive(Debug, Serialize)]
pub struct ConvergenceAnalysis {
    pub concept: String,
    pub models_tested: Vec<String>,
    pub convergence_score: f64,
    pub semantic_overlap: HashMap<String, f64>,
    pub tower_level: usize, // Babel tower level
}

pub struct TraceRecorder {
    traces: Vec<AITrace>,
    output_dir: String,
}

impl TraceRecorder {
    pub fn new(output_dir: &str) -> Self {
        fs::create_dir_all(output_dir).ok();
        Self {
            traces: Vec::new(),
            output_dir: output_dir.to_string(),
        }
    }
    
    pub fn record(&mut self, trace: AITrace) {
        self.traces.push(trace);
    }
    
    pub fn save(&self) -> Result<()> {
        let path = format!("{}/full_trace.json", self.output_dir);
        fs::write(path, serde_json::to_string_pretty(&self.traces)?)?;
        Ok(())
    }
    
    pub fn analyze_convergence(&self, concept: &str) -> ConvergenceAnalysis {
        let relevant: Vec<_> = self.traces.iter()
            .filter(|t| t.output.concepts_extracted.contains(&concept.to_string()))
            .collect();
        
        let models: Vec<_> = relevant.iter()
            .map(|t| t.model.clone())
            .collect();
        
        // Calculate semantic overlap
        let mut overlap = HashMap::new();
        for trace in &relevant {
            for concept in &trace.output.concepts_extracted {
                *overlap.entry(concept.clone()).or_insert(0.0) += 1.0;
            }
        }
        
        let total = relevant.len() as f64;
        for val in overlap.values_mut() {
            *val /= total;
        }
        
        ConvergenceAnalysis {
            concept: concept.to_string(),
            models_tested: models,
            convergence_score: overlap.values().sum::<f64>() / overlap.len() as f64,
            semantic_overlap: overlap,
            tower_level: 0,
        }
    }
}

/// Harmonic semantic filter
pub struct HarmonicFilter {
    pub base_freq: f64, // 432 Hz
    pub active_primes: Vec<u32>,
}

impl HarmonicFilter {
    pub fn new() -> Self {
        Self {
            base_freq: 432.0,
            active_primes: vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71],
        }
    }
    
    /// Remove frequency class (prime) - creates Tower of Babel
    pub fn remove_class(&mut self, prime: u32) {
        self.active_primes.retain(|&p| p != prime);
    }
    
    /// Check if trace belongs to active frequency classes
    pub fn filter(&self, trace: &AITrace) -> bool {
        if let Some(class) = trace.input.harmonic_class {
            self.active_primes.contains(&class)
        } else {
            true
        }
    }
    
    /// Calculate semantic lattice level
    pub fn lattice_level(&self) -> usize {
        15 - self.active_primes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_harmonic_filter() {
        let mut filter = HarmonicFilter::new();
        assert_eq!(filter.lattice_level(), 0);
        
        filter.remove_class(2);
        assert_eq!(filter.lattice_level(), 1);
        
        filter.remove_class(3);
        assert_eq!(filter.lattice_level(), 2);
    }
}
