// 71-layer autoencoder for WASM - Monster architecture

use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

// Monster prime architecture: 5→11→23→47→71→47→23→11→5
const LAYER_SIZES: [usize; 9] = [5, 11, 23, 47, 71, 47, 23, 11, 5];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub size: usize,
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
}

#[wasm_bindgen]
pub struct MonsterAutoencoder {
    layers: Vec<Layer>,
}

#[wasm_bindgen]
impl MonsterAutoencoder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        log("🔮 Initializing 71-layer Monster Autoencoder");
        
        let mut layers = Vec::new();
        
        // Build encoder + decoder
        for i in 0..LAYER_SIZES.len() - 1 {
            let input_size = LAYER_SIZES[i];
            let output_size = LAYER_SIZES[i + 1];
            
            layers.push(Layer {
                size: output_size,
                weights: vec![0.1; input_size * output_size],
                biases: vec![0.0; output_size],
            });
        }
        
        log(&format!("✅ {} layers initialized", layers.len()));
        
        Self { layers }
    }
    
    #[wasm_bindgen]
    pub fn encode_binary(&self, binary: &[u8]) -> Vec<f32> {
        // Binary → 5D
        let mut activation: Vec<f32> = binary.iter()
            .take(5)
            .map(|&b| b as f32 / 255.0)
            .collect();
        
        // Pad to 5D
        while activation.len() < 5 {
            activation.push(0.0);
        }
        
        // Walk up: 5→11→23→47→71
        for i in 0..4 {
            activation = self.forward_layer(i, &activation);
        }
        
        activation
    }
    
    #[wasm_bindgen]
    pub fn decode_to_binary(&self, latent: &[f32]) -> Vec<u8> {
        let mut activation = latent.to_vec();
        
        // Walk down: 71→47→23→11→5
        for i in 4..8 {
            activation = self.forward_layer(i, &activation);
        }
        
        // 5D → Binary
        activation.iter()
            .map(|&v| (v.max(0.0).min(1.0) * 255.0) as u8)
            .collect()
    }
    
    #[wasm_bindgen]
    pub fn walk_up(&self, input: Vec<f32>) -> String {
        let mut activation = input;
        let mut walk = Vec::new();
        
        walk.push(format!("Input: 5D {:?}", &activation[..activation.len().min(5)]));
        
        for (i, size) in [11, 23, 47, 71].iter().enumerate() {
            activation = self.forward_layer(i, &activation);
            walk.push(format!("Layer {}: {}D (mean: {:.3})", 
                i + 1, size, activation.iter().sum::<f32>() / activation.len() as f32));
        }
        
        walk.join("\n")
    }
    
    #[wasm_bindgen]
    pub fn walk_down(&self, latent: Vec<f32>) -> String {
        let mut activation = latent;
        let mut walk = Vec::new();
        
        walk.push(format!("Latent: 71D (mean: {:.3})", 
            activation.iter().sum::<f32>() / activation.len() as f32));
        
        for (i, size) in [47, 23, 11, 5].iter().enumerate() {
            activation = self.forward_layer(4 + i, &activation);
            walk.push(format!("Layer {}: {}D (mean: {:.3})", 
                5 + i, size, activation.iter().sum::<f32>() / activation.len() as f32));
        }
        
        walk.join("\n")
    }
    
    fn forward_layer(&self, layer_idx: usize, input: &[f32]) -> Vec<f32> {
        let layer = &self.layers[layer_idx];
        let mut output = vec![0.0; layer.size];
        
        for i in 0..layer.size {
            let mut sum = layer.biases[i];
            for j in 0..input.len() {
                sum += input[j] * layer.weights[j * layer.size + i];
            }
            output[i] = sum.tanh(); // Activation
        }
        
        output
    }
    
    #[wasm_bindgen]
    pub fn get_layer_sizes(&self) -> Vec<usize> {
        LAYER_SIZES.to_vec()
    }
}

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}
