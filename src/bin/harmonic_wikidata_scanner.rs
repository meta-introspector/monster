// Rust: Harmonic & inductive scanner for Wikidata QIDs

use std::collections::{HashMap, HashSet};

/// Wikidata ontology class
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OntologyClass {
    Entity,           // Q35120 (root)
    Person,           // Q5
    Place,            // Q17334923
    Organization,     // Q43229
    Work,             // Q386724
    Concept,          // Q151885
    Event,            // Q1656682
    Species,          // Q7432
    ChemicalCompound, // Q11173
    AstronomicalObject, // Q6999
}

impl OntologyClass {
    /// Get prime resonance for class
    pub fn prime(&self) -> u32 {
        match self {
            OntologyClass::Entity => 2,
            OntologyClass::Person => 3,
            OntologyClass::Place => 5,
            OntologyClass::Organization => 7,
            OntologyClass::Work => 11,
            OntologyClass::Concept => 13,
            OntologyClass::Event => 17,
            OntologyClass::Species => 19,
            OntologyClass::ChemicalCompound => 23,
            OntologyClass::AstronomicalObject => 29,
        }
    }
    
    /// Harmonic frequency
    pub fn frequency(&self) -> f32 {
        let p = self.prime() as f32;
        440.0 * 2.0_f32.powf(p.ln() / 12.0)
    }
}

/// Wikidata QID with ontology
#[derive(Debug, Clone)]
pub struct WikidataEntity {
    pub qid: u64,
    pub label: String,
    pub class: OntologyClass,
    pub parent_qids: Vec<u64>,
    pub child_qids: Vec<u64>,
}

/// Harmonic scanner
pub struct HarmonicScanner {
    entities: HashMap<u64, WikidataEntity>,
    class_index: HashMap<OntologyClass, Vec<u64>>,
}

impl HarmonicScanner {
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            class_index: HashMap::new(),
        }
    }
    
    /// Import Wikidata ontology
    pub fn import_ontology(&mut self, entities: Vec<WikidataEntity>) {
        for entity in entities {
            let qid = entity.qid;
            let class = entity.class.clone();
            
            self.entities.insert(qid, entity);
            self.class_index.entry(class)
                .or_insert_with(Vec::new)
                .push(qid);
        }
    }
    
    /// Harmonic scan: Find entities resonating with frequency
    pub fn harmonic_scan(&self, target_freq: f32, threshold: f32) -> Vec<u64> {
        let mut results = Vec::new();
        
        for (qid, entity) in &self.entities {
            let entity_freq = entity.class.frequency();
            if (entity_freq - target_freq).abs() < threshold {
                results.push(*qid);
            }
        }
        
        results
    }
    
    /// Inductive scan: Find entities by class hierarchy
    pub fn inductive_scan(&self, start_qid: u64, max_depth: usize) -> HashSet<u64> {
        let mut visited = HashSet::new();
        let mut queue = vec![(start_qid, 0)];
        
        while let Some((qid, depth)) = queue.pop() {
            if depth > max_depth || visited.contains(&qid) {
                continue;
            }
            
            visited.insert(qid);
            
            if let Some(entity) = self.entities.get(&qid) {
                // Add children (inductive step)
                for &child in &entity.child_qids {
                    queue.push((child, depth + 1));
                }
            }
        }
        
        visited
    }
    
    /// Combined: Harmonic + inductive scan
    pub fn harmonic_inductive_scan(
        &self,
        class: OntologyClass,
        max_depth: usize,
    ) -> Vec<u64> {
        let mut results = HashSet::new();
        
        // Start with all entities of this class
        if let Some(qids) = self.class_index.get(&class) {
            for &qid in qids {
                // Inductive scan from each
                let descendants = self.inductive_scan(qid, max_depth);
                results.extend(descendants);
            }
        }
        
        results.into_iter().collect()
    }
    
    /// Map to Monster shard by harmonic resonance
    pub fn map_to_shard(&self, qid: u64) -> u8 {
        if let Some(entity) = self.entities.get(&qid) {
            let prime = entity.class.prime();
            (prime % 15) as u8
        } else {
            (qid % 15) as u8
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ontology_primes() {
        assert_eq!(OntologyClass::Person.prime(), 3);
        assert_eq!(OntologyClass::Place.prime(), 5);
    }
    
    #[test]
    fn test_harmonic_scan() {
        let mut scanner = HarmonicScanner::new();
        
        scanner.import_ontology(vec![
            WikidataEntity {
                qid: 5,
                label: "Human".to_string(),
                class: OntologyClass::Person,
                parent_qids: vec![],
                child_qids: vec![],
            },
        ]);
        
        let freq = OntologyClass::Person.frequency();
        let results = scanner.harmonic_scan(freq, 10.0);
        
        assert!(results.contains(&5));
    }
}

fn main() {
    println!("🎵 Harmonic & Inductive Wikidata Scanner");
    println!("="*70);
    println!();
    
    let mut scanner = HarmonicScanner::new();
    
    // Example ontology
    let entities = vec![
        WikidataEntity {
            qid: 5,
            label: "Human".to_string(),
            class: OntologyClass::Person,
            parent_qids: vec![],
            child_qids: vec![42, 100],
        },
        WikidataEntity {
            qid: 42,
            label: "Douglas Adams".to_string(),
            class: OntologyClass::Person,
            parent_qids: vec![5],
            child_qids: vec![],
        },
    ];
    
    scanner.import_ontology(entities);
    
    println!("Ontology classes & harmonics:");
    for class in [OntologyClass::Person, OntologyClass::Place, OntologyClass::Work] {
        println!("  {:?}: Prime {}, Freq {:.2} Hz",
                 class, class.prime(), class.frequency());
    }
    
    println!();
    println!("Harmonic scan (Person frequency):");
    let freq = OntologyClass::Person.frequency();
    let results = scanner.harmonic_scan(freq, 10.0);
    println!("  Found {} entities", results.len());
    
    println!();
    println!("Inductive scan (from Q5):");
    let descendants = scanner.inductive_scan(5, 2);
    println!("  Found {} descendants", descendants.len());
    
    println!();
    println!("="*70);
    println!("✅ Harmonic + inductive scanning ready!");
}
