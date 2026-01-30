// Interactive WASM tool: Value Lattice ↔ Qwen Shard Matcher

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
    
    fn alert(s: &str);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueMatch {
    value: String,
    godel_number: u64,
    prime: u32,
    layer: u32,
    resonance_score: f32,
}

#[wasm_bindgen]
pub struct LatticeMatcher {
    values: HashMap<String, u64>,
    primes: Vec<u32>,
    current_layer: u32,
}

#[wasm_bindgen]
impl LatticeMatcher {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        log("🔮 Lattice Matcher initialized");
        Self {
            values: HashMap::new(),
            primes: vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71],
            current_layer: 0,
        }
    }
    
    #[wasm_bindgen]
    pub fn add_value(&mut self, value: String, godel: u64) {
        self.values.insert(value, godel);
    }
    
    #[wasm_bindgen]
    pub fn match_with_prime(&self, value: &str, prime: u32) -> f32 {
        if let Ok(v) = value.parse::<u32>() {
            let resonance = if v % prime == 0 {
                1.0
            } else {
                1.0 / ((v % prime) as f32 + 1.0)
            };
            resonance
        } else {
            0.0
        }
    }
    
    #[wasm_bindgen]
    pub fn find_matches(&self, target_value: &str) -> String {
        let mut matches = Vec::new();
        
        for (layer, &prime) in self.primes.iter().enumerate() {
            let resonance = self.match_with_prime(target_value, prime);
            if resonance > 0.5 {
                matches.push(format!(
                    "Layer {}: Prime {} (resonance: {:.2})",
                    layer, prime, resonance
                ));
            }
        }
        
        matches.join("\n")
    }
    
    #[wasm_bindgen]
    pub fn set_layer(&mut self, layer: u32) {
        self.current_layer = layer % 71;
        log(&format!("📍 Layer set to {}", self.current_layer));
    }
    
    #[wasm_bindgen]
    pub fn get_prime_for_layer(&self, layer: u32) -> u32 {
        self.primes[(layer as usize) % self.primes.len()]
    }
    
    #[wasm_bindgen]
    pub fn compute_resonance_field(&self, value: &str) -> Vec<f32> {
        let mut field = Vec::new();
        for &prime in &self.primes {
            field.push(self.match_with_prime(value, prime));
        }
        field
    }
}

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}
