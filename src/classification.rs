// Witness classification via Monster symmetries

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WitnessClassification {
    pub witness_id: String,
    pub source: String,
    pub primary_frequency: f64,
    pub harmonic_spectrum: Vec<f64>,
    pub resonance_pattern: [u8; 15],
    pub conjugacy_class: String,
    pub centralizer_order: u128,
    pub symmetry_type: SymmetryType,
    pub shard_count: usize,
    pub shard_entropy: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SymmetryType {
    BinaryMoon,      // 2,3,5,7,11
    WaveCrest,       // 13,17,19,23,29
    DeepResonance,   // 31,41,47,59,71
    Hybrid(Vec<u8>),
}

const MONSTER_FREQUENCIES: [(u64, f64); 15] = [
    (2, 262.0), (3, 317.0), (5, 497.0), (7, 291.0), (11, 838.0),
    (13, 722.0), (17, 402.0), (19, 438.0), (23, 462.0), (29, 612.0),
    (31, 262.0), (41, 506.0), (47, 245.0), (59, 728.0), (71, 681.0),
];

const MONSTER_ORDER: u128 = 808017424794512875886459904961710757005754368000000000;

pub fn classify_witness(witness: &[u8], id: String, source: String) -> WitnessClassification {
    let spectrum = compute_spectrum(witness);
    let (idx, &freq) = spectrum.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
    
    let pattern: [u8; 15] = spectrum.iter()
        .map(|&f| if f > freq * 0.5 { 1 } else { 0 })
        .collect::<Vec<_>>().try_into().unwrap();
    
    let sym_type = match idx {
        0..=4 => SymmetryType::BinaryMoon,
        5..=9 => SymmetryType::WaveCrest,
        _ => SymmetryType::DeepResonance,
    };
    
    WitnessClassification {
        witness_id: id,
        source,
        primary_frequency: freq,
        harmonic_spectrum: spectrum,
        resonance_pattern: pattern,
        conjugacy_class: compute_class(witness),
        centralizer_order: MONSTER_ORDER / count_symmetries(witness),
        symmetry_type: sym_type,
        shard_count: 71,
        shard_entropy: 4.19,
    }
}

fn compute_spectrum(witness: &[u8]) -> Vec<f64> {
    MONSTER_FREQUENCIES.iter().map(|(prime, freq)| {
        let resonance = witness.iter().enumerate()
            .filter(|(i, _)| (i + 1) as u64 % prime == 0)
            .count() as f64 / witness.len() as f64;
        resonance * freq
    }).collect()
}

fn compute_class(witness: &[u8]) -> String {
    let fp: u128 = witness.iter().enumerate()
        .map(|(i, &b)| (b as u128) << (i % 128))
        .fold(0, |acc, x| acc ^ x);
    format!("{}A", fp % 194)
}

fn count_symmetries(witness: &[u8]) -> u128 {
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]
        .iter()
        .filter(|&&p| is_symmetric(witness, p))
        .product()
}

fn is_symmetric(witness: &[u8], prime: u64) -> bool {
    let n = witness.len();
    (0..n).all(|i| witness[i] == witness[(i + prime as usize) % n])
}
