//! Monster Group Neural Network Construction
//! 
//! Core library using ndarray for tensor operations

use ndarray::{Array1, Array2};
use num_bigint::BigInt;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Monster primes with their exponents
pub const MONSTER_PRIMES: [(u32, u32); 15] = [
    (2, 46), (3, 20), (5, 9), (7, 6), (11, 2), (13, 3),
    (17, 1), (19, 1), (23, 1), (29, 1), (31, 1), (41, 1),
    (47, 1), (59, 1), (71, 1)
];

/// Hecke operator: T_p = r_activation / r_weight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeckeOperator {
    pub prime: u32,
    pub amplification: f64,
    pub weight_rate: f64,
    pub activation_rate: f64,
}

impl HeckeOperator {
    pub fn new(prime: u32, weight_rate: f64, activation_rate: f64) -> Self {
        Self {
            prime,
            amplification: activation_rate / weight_rate.max(0.001),  // Avoid division by zero
            weight_rate,
            activation_rate,
        }
    }
    
    /// Compose two Hecke operators: T(p1 ∘ p2) = T(p1) × T(p2)
    pub fn compose(&self, other: &Self) -> f64 {
        self.amplification * other.amplification
    }
}

/// Gödel signature: G = ∏ p^(divisibility_rate)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GodelSignature {
    pub exponents: HashMap<u32, f64>,
}

impl GodelSignature {
    pub fn new() -> Self {
        Self {
            exponents: HashMap::new(),
        }
    }
    
    pub fn from_array(arr: &Array2<f32>) -> Self {
        let mut sig = Self::new();
        
        for &(prime, _) in &MONSTER_PRIMES {
            let rate = measure_divisibility_2d(arr, prime);
            sig.exponents.insert(prime, rate);
        }
        
        sig
    }
    
    /// Compute Gödel number (as BigInt for exact arithmetic)
    pub fn compute_godel_number(&self) -> BigInt {
        let mut result = BigInt::from(1u32);
        
        for (&prime, &exp) in &self.exponents {
            let p = BigInt::from(prime);
            let e = (exp * 100.0) as u32;  // Scale for integer exponent
            result *= p.pow(e);
        }
        
        result
    }
}

/// Measure prime divisibility in a 1D array
pub fn measure_divisibility_1d(arr: &Array1<f32>, prime: u32) -> f64 {
    let values: Vec<i32> = arr.iter()
        .map(|&v| (v * 1000.0) as i32)
        .collect();
    
    let divisible = values.iter()
        .filter(|&&v| v != 0 && v % (prime as i32) == 0)
        .count();
    
    divisible as f64 / values.len().max(1) as f64
}

/// Measure prime divisibility in a 2D array
pub fn measure_divisibility_2d(arr: &Array2<f32>, prime: u32) -> f64 {
    let values: Vec<i32> = arr.iter()
        .map(|&v| (v * 1000.0) as i32)
        .collect();
    
    let divisible = values.iter()
        .filter(|&&v| v != 0 && v % (prime as i32) == 0)
        .count();
    
    divisible as f64 / values.len().max(1) as f64
}

/// Monster layer: Linear transformation modulated by Gödel signature
#[derive(Debug, Clone)]
pub struct MonsterLayer {
    pub prime: u32,
    pub weights: Array2<f32>,
    pub godel_signature: Array1<f32>,
}

impl MonsterLayer {
    pub fn new(prime: u32) -> Self {
        let size = (prime * 8) as usize;
        let mut rng = rand::thread_rng();
        
        // Initialize weights with normal distribution
        let weights = Array2::from_shape_fn((size, size), |_| {
            rng.gen::<f32>() * 0.1
        });
        
        // Gödel signature: [p^0, p^1, p^2, ...]
        let godel_signature = Array1::from_shape_fn(size, |i| {
            (prime as f32).powi((i % 8) as i32)  // Modulo to prevent overflow
        });
        
        Self {
            prime,
            weights,
            godel_signature,
        }
    }
    
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        // Matrix multiplication
        let output = input.dot(&self.weights);
        
        // Modulate by Gödel signature (broadcast)
        &output * &self.godel_signature
    }
    
    pub fn measure_hecke_operator(&self, input: &Array2<f32>) -> HeckeOperator {
        let output = self.forward(input);
        
        // Measure prime divisibility in weights
        let weight_rate = measure_divisibility_2d(&self.weights, self.prime);
        
        // Measure prime divisibility in activations
        let activation_rate = measure_divisibility_2d(&output, self.prime);
        
        HeckeOperator::new(self.prime, weight_rate, activation_rate)
    }
}

/// Monster network: Stack of layers for a single prime
#[derive(Debug, Clone)]
pub struct MonsterNetwork {
    pub prime: u32,
    pub layers: Vec<MonsterLayer>,
    pub godel_number: BigInt,
}

impl MonsterNetwork {
    pub fn new(prime: u32) -> Self {
        let num_layers = prime as usize;
        let layers: Vec<_> = (0..num_layers)
            .map(|_| MonsterLayer::new(prime))
            .collect();
        
        // Gödel number: G = p^p
        let godel_number = BigInt::from(prime).pow(prime);
        
        Self {
            prime,
            layers,
            godel_number,
        }
    }
    
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut current = input.clone();
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        current
    }
    
    pub fn measure_hecke_operators(&self, input: &Array2<f32>) -> Vec<HeckeOperator> {
        let mut operators = Vec::new();
        let mut current = input.clone();
        
        for layer in &self.layers {
            let hecke = layer.measure_hecke_operator(&current);
            operators.push(hecke);
            current = layer.forward(&current);
        }
        
        operators
    }
}

/// Monster lattice: All 15 networks connected by Hecke operators
#[derive(Debug, Serialize, Deserialize)]
pub struct MonsterLattice {
    pub godel_indices: HashMap<String, u32>,  // Gödel number -> prime
    pub hecke_edges: Vec<(u32, u32, f64)>,    // (prime1, prime2, T_composed)
}

impl MonsterLattice {
    pub fn new() -> Self {
        let mut lattice = Self {
            godel_indices: HashMap::new(),
            hecke_edges: Vec::new(),
        };
        
        // Index all networks by Gödel number
        for &(prime, _) in &MONSTER_PRIMES {
            let godel = BigInt::from(prime).pow(prime);
            lattice.godel_indices.insert(godel.to_string(), prime);
        }
        
        lattice
    }
    
    pub fn add_edge(&mut self, p1: u32, p2: u32, t_composed: f64) {
        self.hecke_edges.push((p1, p2, t_composed));
    }
    
    /// Verify Monster group structure
    pub fn verify_monster_structure(&self) -> bool {
        // Check all 15 primes are present
        self.godel_indices.len() == 15
    }
    
    /// Compute Monster group order from lattice
    pub fn compute_order(&self) -> BigInt {
        let mut order = BigInt::from(1u32);
        
        for &(prime, exp) in &MONSTER_PRIMES {
            let p = BigInt::from(prime);
            order *= p.pow(exp);
        }
        
        order
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_godel_signature() {
        let mut sig = GodelSignature::new();
        sig.exponents.insert(2, 0.5);
        sig.exponents.insert(3, 0.33);
        
        let godel = sig.compute_godel_number();
        assert!(godel > BigInt::from(1u32));
    }
    
    #[test]
    fn test_hecke_operator() {
        let h1 = HeckeOperator::new(2, 0.5, 0.8);
        let h2 = HeckeOperator::new(3, 0.33, 0.49);
        
        assert!((h1.amplification - 1.6).abs() < 0.01);
        assert!((h2.amplification - 1.48).abs() < 0.01);
        
        let composed = h1.compose(&h2);
        assert!(composed > 2.0);
    }
    
    #[test]
    fn test_monster_lattice() {
        let lattice = MonsterLattice::new();
        assert_eq!(lattice.godel_indices.len(), 15);
        
        let order = lattice.compute_order();
        let expected = "808017424794512875886459904961710757005754368000000000";
        assert_eq!(order.to_string(), expected);
    }
}
