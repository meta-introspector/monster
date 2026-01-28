use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[derive(Serialize, Deserialize, Clone)]
pub struct MonsterGroup {
    pub group_number: usize,
    pub position: usize,
    pub sequence: String,
    pub digits: usize,
    pub factors_removed: usize,
    pub removed_primes: Vec<u32>,
}

#[wasm_bindgen]
pub struct MonsterWalk {
    groups: Vec<MonsterGroup>,
}

#[wasm_bindgen]
impl MonsterWalk {
    #[wasm_bindgen(constructor)]
    pub fn new() -> MonsterWalk {
        log("Initializing Monster Walk WASM module");
        
        let groups = vec![
            MonsterGroup {
                group_number: 1,
                position: 0,
                sequence: "8080".to_string(),
                digits: 4,
                factors_removed: 8,
                removed_primes: vec![7, 11, 17, 19, 29, 31, 41, 59],
            },
            MonsterGroup {
                group_number: 2,
                position: 4,
                sequence: "1742".to_string(),
                digits: 4,
                factors_removed: 4,
                removed_primes: vec![3, 5, 13, 31],
            },
            MonsterGroup {
                group_number: 3,
                position: 8,
                sequence: "479".to_string(),
                digits: 3,
                factors_removed: 4,
                removed_primes: vec![3, 13, 31, 71],
            },
            MonsterGroup {
                group_number: 4,
                position: 11,
                sequence: "451".to_string(),
                digits: 3,
                factors_removed: 4,
                removed_primes: vec![5, 7, 19, 41],
            },
            MonsterGroup {
                group_number: 5,
                position: 14,
                sequence: "2875".to_string(),
                digits: 4,
                factors_removed: 4,
                removed_primes: vec![5, 11, 29, 41],
            },
            MonsterGroup {
                group_number: 6,
                position: 18,
                sequence: "8864".to_string(),
                digits: 4,
                factors_removed: 8,
                removed_primes: vec![3, 5, 7, 11, 17, 19, 41, 71],
            },
            MonsterGroup {
                group_number: 7,
                position: 22,
                sequence: "5990".to_string(),
                digits: 4,
                factors_removed: 8,
                removed_primes: vec![2, 3, 13, 31, 41, 47, 59, 71],
            },
            MonsterGroup {
                group_number: 8,
                position: 26,
                sequence: "496".to_string(),
                digits: 3,
                factors_removed: 6,
                removed_primes: vec![2, 7, 19, 31, 47, 71],
            },
            MonsterGroup {
                group_number: 9,
                position: 29,
                sequence: "1710".to_string(),
                digits: 4,
                factors_removed: 3,
                removed_primes: vec![5, 41, 59],
            },
            MonsterGroup {
                group_number: 10,
                position: 33,
                sequence: "7570".to_string(),
                digits: 4,
                factors_removed: 8,
                removed_primes: vec![2, 7, 17, 23, 29, 41, 47, 59],
            },
        ];
        
        MonsterWalk { groups }
    }
    
    #[wasm_bindgen]
    pub fn get_group_count(&self) -> usize {
        self.groups.len()
    }
    
    #[wasm_bindgen]
    pub fn get_group(&self, index: usize) -> JsValue {
        if index < self.groups.len() {
            serde_wasm_bindgen::to_value(&self.groups[index]).unwrap()
        } else {
            JsValue::NULL
        }
    }
    
    #[wasm_bindgen]
    pub fn get_all_groups(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.groups).unwrap()
    }
    
    #[wasm_bindgen]
    pub fn compute_harmonic(&self, prime: u32, exponent: u32) -> f64 {
        432.0 * (prime as f64) * (exponent as f64)
    }
    
    #[wasm_bindgen]
    pub fn get_bott_period_groups(&self) -> Vec<usize> {
        self.groups.iter()
            .filter(|g| g.factors_removed == 8)
            .map(|g| g.group_number)
            .collect()
    }
    
    #[wasm_bindgen]
    pub fn get_monster_order(&self) -> String {
        "808017424794512875886459904961710757005754368000000000".to_string()
    }
}

#[wasm_bindgen(start)]
pub fn main() {
    log("Monster Walk WASM module loaded!");
}
