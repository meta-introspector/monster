// Monster Prime Introspector for mistral.rs
// Hooks into model inference to analyze weights and activations

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use serde::Serialize;

const MONSTER_PRIMES: [u32; 15] = [2,3,5,7,11,13,17,19,23,29,31,41,47,59,71];

lazy_static::lazy_static! {
    static ref MONSTER_TRACE: Arc<Mutex<MonsterTrace>> = Arc::new(Mutex::new(MonsterTrace::new()));
}

#[derive(Debug, Serialize)]
pub struct MonsterTrace {
    layers: Vec<LayerTrace>,
    current_layer: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct LayerTrace {
    name: String,
    weight_primes: HashMap<u32, f64>,
    activation_primes: HashMap<u32, f64>,
    timestamp: std::time::Instant,
}

impl MonsterTrace {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            current_layer: None,
        }
    }
    
    pub fn enter_layer(&mut self, name: &str) {
        self.current_layer = Some(name.to_string());
    }
    
    pub fn record_weights(&mut self, name: &str, weights: &[f32]) {
        let primes = analyze_float_primes(weights);
        
        if let Some(layer) = self.layers.iter_mut().find(|l| l.name == name) {
            layer.weight_primes = primes;
        } else {
            self.layers.push(LayerTrace {
                name: name.to_string(),
                weight_primes: primes,
                activation_primes: HashMap::new(),
                timestamp: std::time::Instant::now(),
            });
        }
    }
    
    pub fn record_activations(&mut self, name: &str, activations: &[f32]) {
        let primes = analyze_float_primes(activations);
        
        if let Some(layer) = self.layers.iter_mut().find(|l| l.name == name) {
            layer.activation_primes = primes;
        }
    }
    
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

pub struct MonsterGuard {
    layer_name: String,
    start: std::time::Instant,
}

impl MonsterGuard {
    pub fn enter(name: &str) -> Self {
        MONSTER_TRACE.lock().unwrap().enter_layer(name);
        
        Self {
            layer_name: name.to_string(),
            start: std::time::Instant::now(),
        }
    }
    
    pub fn exit_with_result<T>(&self, _result: &T) {
        let elapsed = self.start.elapsed();
        println!("  {} took {:?}", self.layer_name, elapsed);
    }
}

/// Analyze float values for Monster prime patterns
/// Quantize to int32 and check divisibility
pub fn analyze_float_primes(values: &[f32]) -> HashMap<u32, f64> {
    let mut prime_counts = HashMap::new();
    
    // Quantize floats to int32 (scale by 10000)
    let quantized: Vec<i32> = values.iter()
        .map(|&v| (v * 10000.0) as i32)
        .collect();
    
    for &prime in &MONSTER_PRIMES {
        let divisible = quantized.iter()
            .filter(|&&v| v.abs() % prime as i32 == 0)
            .count();
        
        let percentage = divisible as f64 / quantized.len() as f64;
        prime_counts.insert(prime, percentage);
    }
    
    prime_counts
}

pub fn analyze_weights(layer_name: &str, weights: &[f32]) {
    let mut trace = MONSTER_TRACE.lock().unwrap();
    trace.record_weights(layer_name, weights);
    
    let primes = analyze_float_primes(weights);
    println!("  Weights {}: P2={:.1}%, P3={:.1}%, P5={:.1}%",
             layer_name,
             primes.get(&2).unwrap_or(&0.0) * 100.0,
             primes.get(&3).unwrap_or(&0.0) * 100.0,
             primes.get(&5).unwrap_or(&0.0) * 100.0);
}

pub fn analyze_prime_patterns(activations: &[f32]) -> HashMap<u32, f64> {
    let primes = analyze_float_primes(activations);
    
    if let Some(layer) = MONSTER_TRACE.lock().unwrap().current_layer.clone() {
        MONSTER_TRACE.lock().unwrap().record_activations(&layer, activations);
    }
    
    primes
}

pub fn save_trace(path: &str) -> std::io::Result<()> {
    MONSTER_TRACE.lock().unwrap().save(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_prime_analysis() {
        let values = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let primes = analyze_float_primes(&values);
        
        // All even → high prime 2
        assert!(primes[&2] > 0.9);
    }
}
