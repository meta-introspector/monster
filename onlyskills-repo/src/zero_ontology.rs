// Zero Ontology via Monster Walk and 10-fold Way
use serde::{Deserialize, Serialize};

// Monster Walk steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MonsterStep {
    Full,           // All 15 primes
    Step1,          // 8080 (remove 8 primes)
    Step2,          // 1742 (remove 4 primes)
    Step3,          // 479 (remove 4 primes)
}

// 10-fold Way symmetry classes (Altland-Zirnbauer)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TenfoldClass {
    A,      // Unitary
    AIII,   // Chiral unitary
    AI,     // Orthogonal
    BDI,    // Chiral orthogonal
    D,      // Orthogonal (no TRS)
    DIII,   // Chiral orthogonal (TRS)
    AII,    // Symplectic
    CII,    // Chiral symplectic
    C,      // Symplectic (no TRS)
    CI,     // Chiral symplectic (TRS)
}

impl TenfoldClass {
    pub fn from_index(i: usize) -> Self {
        match i % 10 {
            0 => TenfoldClass::A,
            1 => TenfoldClass::AIII,
            2 => TenfoldClass::AI,
            3 => TenfoldClass::BDI,
            4 => TenfoldClass::D,
            5 => TenfoldClass::DIII,
            6 => TenfoldClass::AII,
            7 => TenfoldClass::CII,
            8 => TenfoldClass::C,
            9 => TenfoldClass::CI,
            _ => unreachable!(),
        }
    }
}

// Zero point: intersection of Monster Walk and 10-fold Way
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroPoint {
    pub monster_step: MonsterStep,
    pub tenfold_class: TenfoldClass,
    pub coords: [u8; 10],  // 10-dimensional zero
}

impl ZeroPoint {
    pub fn new() -> Self {
        Self {
            monster_step: MonsterStep::Full,
            tenfold_class: TenfoldClass::A,
            coords: [0; 10],
        }
    }
}

// Intrinsic semantics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntrinsicSemantics {
    pub structure: String,
    pub relations: Vec<String>,
    pub constraints: Vec<String>,
}

// Zero ontology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroOntology {
    pub zero: ZeroPoint,
    pub entity_coords: [u8; 10],
    pub semantics: IntrinsicSemantics,
}

impl ZeroOntology {
    pub fn from_prime(p: u64) -> Self {
        let monster_step = prime_to_monster_step(p);
        let tenfold_class = TenfoldClass::from_index((p % 10) as usize);
        
        let zero = ZeroPoint {
            monster_step,
            tenfold_class,
            coords: [0; 10],
        };
        
        let entity_coords = prime_displacement(p);
        
        let semantics = IntrinsicSemantics {
            structure: format!("prime({})", p),
            relations: vec!["divides".to_string(), "factors".to_string()],
            constraints: vec![format!("{} > 0", p), "is_prime".to_string()],
        };
        
        Self {
            zero,
            entity_coords,
            semantics,
        }
    }
    
    pub fn from_genus(g: u8) -> Self {
        let monster_step = genus_to_monster_step(g);
        let tenfold_class = TenfoldClass::from_index(g as usize);
        
        let zero = ZeroPoint {
            monster_step,
            tenfold_class,
            coords: [0; 10],
        };
        
        let entity_coords = genus_displacement(g);
        
        let semantics = IntrinsicSemantics {
            structure: format!("genus({})", g),
            relations: vec!["modular_curve".to_string(), "cusps".to_string()],
            constraints: vec![format!("{} >= 0", g)],
        };
        
        Self {
            zero,
            entity_coords,
            semantics,
        }
    }
    
    pub fn path_from_zero(&self) -> Vec<[u8; 10]> {
        // Path from zero to entity
        let mut path = Vec::new();
        
        for i in 0..10 {
            let mut step = [0u8; 10];
            for j in 0..=i {
                step[j] = self.entity_coords[j];
            }
            path.push(step);
        }
        
        path
    }
}

fn prime_to_monster_step(p: u64) -> MonsterStep {
    const MONSTER_PRIMES: [u64; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];
    const STEP1_REMOVED: [u64; 8] = [7, 11, 17, 19, 29, 31, 41, 59];
    const STEP2_REMOVED: [u64; 4] = [3, 5, 13, 31];
    const STEP3_REMOVED: [u64; 4] = [3, 13, 31, 71];
    
    if MONSTER_PRIMES.contains(&p) {
        if STEP3_REMOVED.contains(&p) {
            MonsterStep::Step3
        } else if STEP2_REMOVED.contains(&p) {
            MonsterStep::Step2
        } else if STEP1_REMOVED.contains(&p) {
            MonsterStep::Step1
        } else {
            MonsterStep::Full
        }
    } else {
        MonsterStep::Full
    }
}

fn genus_to_monster_step(g: u8) -> MonsterStep {
    match g {
        0 => MonsterStep::Full,
        1..=2 => MonsterStep::Step1,
        3..=4 => MonsterStep::Step2,
        _ => MonsterStep::Step3,
    }
}

fn prime_displacement(p: u64) -> [u8; 10] {
    let mut disp = [0u8; 10];
    for i in 0..10 {
        disp[i] = (p % 71) as u8;
    }
    disp
}

fn genus_displacement(g: u8) -> [u8; 10] {
    let mut disp = [0u8; 10];
    for i in 0..10 {
        disp[i] = (g * 2) % 71;
    }
    disp
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_zero_ontology_prime() {
        let onto = ZeroOntology::from_prime(71);
        
        assert_eq!(onto.entity_coords.len(), 10);
        assert_eq!(onto.semantics.structure, "prime(71)");
    }
    
    #[test]
    fn test_zero_ontology_genus() {
        let onto = ZeroOntology::from_genus(0);
        
        assert_eq!(onto.entity_coords.len(), 10);
        assert_eq!(onto.semantics.structure, "genus(0)");
    }
    
    #[test]
    fn test_path_from_zero() {
        let onto = ZeroOntology::from_prime(71);
        let path = onto.path_from_zero();
        
        assert_eq!(path.len(), 10);
        assert_eq!(path[0][0], onto.entity_coords[0]);
    }
}
