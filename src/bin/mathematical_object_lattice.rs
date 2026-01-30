// Mathematical Object Lattice: From simple groups to Monster
// Includes: Sporadic groups, Leech lattice, Umbral moonshine

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ============================================================================
// LATTICE STRUCTURE
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum LatticeType {
    Leech,      // 24-dimensional Leech lattice
    Umbral,     // Umbral moonshine lattices
    E8,         // E8 lattice
    Standard,   // Standard integer lattice
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum ObjectType {
    SporadicGroup,
    SimpleGroup,
    LieGroup,
    Lattice,
    ModularForm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
struct Complexity(u64);

// ============================================================================
// MATHEMATICAL OBJECT
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MathematicalObject {
    name: String,
    object_type: ObjectType,
    lattice_type: Option<LatticeType>,
    complexity: Complexity,
    factorization: Vec<(u32, u32)>,  // prime^exponent
    dimension: Option<usize>,
    parent: Option<String>,
}

impl MathematicalObject {
    fn order(&self) -> Option<u128> {
        if self.factorization.is_empty() {
            return None;
        }
        
        let mut order: u128 = 1;
        for (prime, exp) in &self.factorization {
            order = order.saturating_mul((*prime as u128).pow(*exp));
        }
        Some(order)
    }
    
    fn num_shards(&self) -> usize {
        self.factorization.len()
    }
    
    fn max_layers(&self) -> usize {
        self.factorization.iter().map(|(_, exp)| *exp as usize).max().unwrap_or(1)
    }
    
    fn primes(&self) -> Vec<u32> {
        self.factorization.iter().map(|(p, _)| *p).collect()
    }
}

// ============================================================================
// OBJECT CATALOG
// ============================================================================

struct ObjectLattice {
    objects: HashMap<String, MathematicalObject>,
}

impl ObjectLattice {
    fn new() -> Self {
        let mut lattice = Self {
            objects: HashMap::new(),
        };
        lattice.populate_sporadic_groups();
        lattice.populate_simple_groups();
        lattice.populate_lattices();
        lattice
    }
    
    fn populate_sporadic_groups(&mut self) {
        // Monster group (largest)
        self.objects.insert("Monster".to_string(), MathematicalObject {
            name: "Monster".to_string(),
            object_type: ObjectType::SporadicGroup,
            lattice_type: Some(LatticeType::Leech),
            complexity: Complexity(808017424794512875886459904961710757005754368000000000),
            factorization: vec![
                (2, 46), (3, 20), (5, 9), (7, 6), (11, 2),
                (13, 3), (17, 1), (19, 1), (23, 1), (29, 1),
                (31, 1), (41, 1), (47, 1), (59, 1), (71, 1),
            ],
            dimension: Some(196883),
            parent: None,
        });
        
        // Baby Monster
        self.objects.insert("BabyMonster".to_string(), MathematicalObject {
            name: "BabyMonster".to_string(),
            object_type: ObjectType::SporadicGroup,
            lattice_type: Some(LatticeType::Leech),
            complexity: Complexity(4154781481226426191177580544000000),
            factorization: vec![
                (2, 41), (3, 13), (5, 6), (7, 2), (11, 1),
                (13, 1), (17, 1), (19, 1), (23, 1), (31, 1), (47, 1),
            ],
            dimension: Some(4371),
            parent: Some("Monster".to_string()),
        });
        
        // Fischer groups
        self.objects.insert("Fi24".to_string(), MathematicalObject {
            name: "Fi24".to_string(),
            object_type: ObjectType::SporadicGroup,
            lattice_type: Some(LatticeType::Leech),
            complexity: Complexity(1255205709190661721292800),
            factorization: vec![
                (2, 21), (3, 16), (5, 2), (7, 3), (11, 1),
                (13, 1), (17, 1), (23, 1), (29, 1),
            ],
            dimension: Some(8671),
            parent: Some("Monster".to_string()),
        });
        
        // Conway groups
        self.objects.insert("Co1".to_string(), MathematicalObject {
            name: "Co1".to_string(),
            object_type: ObjectType::SporadicGroup,
            lattice_type: Some(LatticeType::Leech),
            complexity: Complexity(4157776806543360000),
            factorization: vec![
                (2, 21), (3, 9), (5, 4), (7, 2), (11, 1), (13, 1), (23, 1),
            ],
            dimension: Some(24),
            parent: Some("Monster".to_string()),
        });
        
        // Mathieu groups
        self.objects.insert("M24".to_string(), MathematicalObject {
            name: "M24".to_string(),
            object_type: ObjectType::SporadicGroup,
            lattice_type: Some(LatticeType::Leech),
            complexity: Complexity(244823040),
            factorization: vec![
                (2, 10), (3, 3), (5, 1), (7, 1), (11, 1), (23, 1),
            ],
            dimension: Some(11),
            parent: Some("Co1".to_string()),
        });
        
        self.objects.insert("M12".to_string(), MathematicalObject {
            name: "M12".to_string(),
            object_type: ObjectType::SporadicGroup,
            lattice_type: None,
            complexity: Complexity(95040),
            factorization: vec![
                (2, 6), (3, 3), (5, 1), (11, 1),
            ],
            dimension: Some(11),
            parent: Some("M24".to_string()),
        });
    }
    
    fn populate_simple_groups(&mut self) {
        // Alternating groups
        self.objects.insert("A5".to_string(), MathematicalObject {
            name: "A5".to_string(),
            object_type: ObjectType::SimpleGroup,
            lattice_type: None,
            complexity: Complexity(60),
            factorization: vec![(2, 2), (3, 1), (5, 1)],
            dimension: Some(3),
            parent: None,
        });
        
        self.objects.insert("A6".to_string(), MathematicalObject {
            name: "A6".to_string(),
            object_type: ObjectType::SimpleGroup,
            lattice_type: None,
            complexity: Complexity(360),
            factorization: vec![(2, 3), (3, 2), (5, 1)],
            dimension: Some(5),
            parent: Some("A5".to_string()),
        });
        
        // Cyclic groups (simplest)
        self.objects.insert("Z2".to_string(), MathematicalObject {
            name: "Z2".to_string(),
            object_type: ObjectType::SimpleGroup,
            lattice_type: None,
            complexity: Complexity(2),
            factorization: vec![(2, 1)],
            dimension: Some(1),
            parent: None,
        });
        
        self.objects.insert("Z3".to_string(), MathematicalObject {
            name: "Z3".to_string(),
            object_type: ObjectType::SimpleGroup,
            lattice_type: None,
            complexity: Complexity(3),
            factorization: vec![(3, 1)],
            dimension: Some(1),
            parent: None,
        });
    }
    
    fn populate_lattices(&mut self) {
        // Leech lattice
        self.objects.insert("Leech".to_string(), MathematicalObject {
            name: "Leech".to_string(),
            object_type: ObjectType::Lattice,
            lattice_type: Some(LatticeType::Leech),
            complexity: Complexity(8315553613086720000),  // |Co0|
            factorization: vec![
                (2, 22), (3, 9), (5, 4), (7, 2), (11, 1), (13, 1), (23, 1),
            ],
            dimension: Some(24),
            parent: None,
        });
        
        // E8 lattice
        self.objects.insert("E8".to_string(), MathematicalObject {
            name: "E8".to_string(),
            object_type: ObjectType::Lattice,
            lattice_type: Some(LatticeType::E8),
            complexity: Complexity(696729600),
            factorization: vec![
                (2, 14), (3, 5), (5, 2), (7, 1),
            ],
            dimension: Some(8),
            parent: None,
        });
        
        // Umbral lattices (23 of them)
        for k in 2..=24 {
            if k == 24 { continue; } // Skip 24 (that's Leech)
            self.objects.insert(format!("Umbral{}", k), MathematicalObject {
                name: format!("Umbral{}", k),
                object_type: ObjectType::Lattice,
                lattice_type: Some(LatticeType::Umbral),
                complexity: Complexity(k as u64),  // Placeholder
                factorization: vec![(k as u32, 1)],
                dimension: Some(k),
                parent: Some("Leech".to_string()),
            });
        }
    }
    
    fn get(&self, name: &str) -> Option<&MathematicalObject> {
        self.objects.get(name)
    }
    
    fn by_complexity(&self) -> Vec<&MathematicalObject> {
        let mut objs: Vec<_> = self.objects.values().collect();
        objs.sort_by_key(|o| o.complexity);
        objs
    }
    
    fn by_type(&self, obj_type: ObjectType) -> Vec<&MathematicalObject> {
        self.objects.values()
            .filter(|o| o.object_type == obj_type)
            .collect()
    }
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    println!("🌌 MATHEMATICAL OBJECT LATTICE");
    println!("{}", "=".repeat(70));
    println!();
    
    let lattice = ObjectLattice::new();
    
    // Display by complexity
    println!("📊 Objects by Complexity:");
    println!("{}", "-".repeat(70));
    
    for obj in lattice.by_complexity() {
        let order_str = obj.order()
            .map(|o| format!("{}", o))
            .unwrap_or_else(|| "N/A".to_string());
        
        let lattice_str = obj.lattice_type
            .map(|l| format!("{:?}", l))
            .unwrap_or_else(|| "None".to_string());
        
        println!("{:20} | {:15?} | {:10} | shards={:2} layers={:2} | {}",
            obj.name,
            obj.object_type,
            lattice_str,
            obj.num_shards(),
            obj.max_layers(),
            if order_str.len() > 30 { &order_str[..30] } else { &order_str }
        );
    }
    
    println!();
    println!("🎯 Sporadic Groups:");
    println!("{}", "-".repeat(70));
    
    for obj in lattice.by_type(ObjectType::SporadicGroup) {
        println!("{:20} | dim={:6?} | parent={:?}",
            obj.name,
            obj.dimension,
            obj.parent
        );
    }
    
    println!();
    println!("🔷 Lattices:");
    println!("{}", "-".repeat(70));
    
    for obj in lattice.by_type(ObjectType::Lattice) {
        if obj.name.starts_with("Umbral") { continue; }
        println!("{:20} | {:?} | dim={:?}",
            obj.name,
            obj.lattice_type,
            obj.dimension
        );
    }
    
    println!();
    println!("🌙 Umbral Moonshine (23 lattices):");
    println!("{}", "-".repeat(70));
    
    let umbral: Vec<_> = lattice.by_type(ObjectType::Lattice)
        .into_iter()
        .filter(|o| o.name.starts_with("Umbral"))
        .collect();
    println!("  {} umbral lattices (dimensions 2-23)", umbral.len());
    
    println!();
    println!("🔥 Monster Group Details:");
    println!("{}", "-".repeat(70));
    
    if let Some(monster) = lattice.get("Monster") {
        println!("Name: {}", monster.name);
        println!("Type: {:?}", monster.object_type);
        println!("Lattice: {:?}", monster.lattice_type);
        println!("Order: {}", monster.order().unwrap());
        println!("Dimension: {:?}", monster.dimension);
        println!("Shards: {}", monster.num_shards());
        println!("Max Layers: {}", monster.max_layers());
        println!("Primes: {:?}", monster.primes());
        println!("Factorization:");
        for (p, e) in &monster.factorization {
            println!("  {}^{}", p, e);
        }
    }
    
    println!();
    println!("✅ Lattice contains {} mathematical objects", lattice.objects.len());
}
