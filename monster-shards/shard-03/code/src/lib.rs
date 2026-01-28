use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct MonsterShard {
    number: u32,
    neurons: Vec<f32>,
}

#[wasm_bindgen]
impl MonsterShard {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            number: 3,
            neurons: vec![],
        }
    }
    
    pub fn get_number(&self) -> u32 {
        self.number
    }
    
    pub fn forward(&self, input: Vec<f32>) -> Vec<f32> {
        // Apply Hecke operator
        input.iter()
            .map(|&x| x * (self.number as f32 / 10.0))
            .collect()
    }
}
