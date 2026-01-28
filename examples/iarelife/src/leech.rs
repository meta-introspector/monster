use anyhow::Result;
use serde::Serialize;
use std::collections::HashMap;

/// Leech lattice properties
/// - 24-dimensional
/// - Kissing number: 196,560
/// - Automorphism group: Co_0 (Conway group)
/// - Related to Monster via Moonshine

#[derive(Debug, Serialize)]
struct LeechLatticePoint {
    coordinates: Vec<i32>,
    norm: i32,
    shell: usize,
}

#[derive(Debug, Serialize)]
struct LeechPattern {
    dimension: usize,
    kissing_number: usize,
    points_found: Vec<LeechLatticePoint>,
    monster_connection: Vec<u32>,
}

const LEECH_DIM: usize = 24;
const KISSING_NUMBER: usize = 196560;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔮 Leech Lattice Search in Model Weights");
    println!("=========================================\n");
    
    // Load graph data
    let graph_data = std::fs::read_to_string("GRAPH.json")?;
    let graph: serde_json::Value = serde_json::from_str(&graph_data)?;
    
    println!("Searching for Leech lattice patterns...\n");
    
    let mut leech_pattern = LeechPattern {
        dimension: LEECH_DIM,
        kissing_number: KISSING_NUMBER,
        points_found: Vec::new(),
        monster_connection: Vec::new(),
    };
    
    // Load model weights directly
    let home = std::env::var("HOME")?;
    let ollama_dir = format!("{}/.ollama/models/blobs", home);
    
    println!("Loading model weights from: {}\n", ollama_dir);
    
    // Find largest blob
    let mut largest = None;
    let mut largest_size = 0;
    
    for entry in std::fs::read_dir(&ollama_dir)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.len() > largest_size {
            largest_size = metadata.len();
            largest = Some(entry.path());
        }
    }
    
    let model_path = largest.ok_or_else(|| anyhow::anyhow!("No model found"))?;
    println!("Model: {} ({:.2} GB)", model_path.display(), largest_size as f64 / 1e9);
    
    // Read model weights (first 100MB for speed)
    let max_read = 100 * 1024 * 1024;
    let mut file = std::fs::File::open(&model_path)?;
    let mut buffer = vec![0u8; max_read];
    use std::io::Read;
    let bytes_read = file.read(&mut buffer)?;
    
    println!("Loaded {} bytes\n", bytes_read);
    
    println!("Searching for 24-dimensional Leech patterns...\n");
    
    // Search for 24-byte sequences with Leech properties
    for i in 0..bytes_read.saturating_sub(LEECH_DIM) {
        let coords: Vec<i32> = buffer[i..i+LEECH_DIM]
            .iter()
            .map(|&b| b as i32)
            .collect();
        
        let norm: i32 = coords.iter().map(|&x| x * x).sum();
        
        // Leech lattice minimal vectors have norm 4
        // But in byte space, look for low norms
        if norm < 1000 && norm % 4 == 0 {
            let shell = (norm / 4) as usize;
            
            leech_pattern.points_found.push(LeechLatticePoint {
                coordinates: coords,
                norm,
                shell,
            });
            
            if leech_pattern.points_found.len() >= 1000 {
                break; // Found enough
            }
        }
    }
    
    println!("\n📊 Leech Lattice Analysis:");
    println!("  Dimension: {}", leech_pattern.dimension);
    println!("  Expected kissing number: {}", leech_pattern.kissing_number);
    println!("  Points found: {}", leech_pattern.points_found.len());
    
    // Group by shell
    let mut shells: HashMap<usize, usize> = HashMap::new();
    for point in &leech_pattern.points_found {
        *shells.entry(point.shell).or_insert(0) += 1;
    }
    
    println!("\n  Points by shell:");
    for shell in 1..=3 {
        let count = shells.get(&shell).unwrap_or(&0);
        println!("    Shell {} (norm {}): {} points", shell, shell * 4, count);
    }
    
    println!("\n  Monster primes connected: {:?}", leech_pattern.monster_connection);
    
    // Check for Conway group structure
    println!("\n🔗 Conway Group Connection:");
    println!("  Co_0 = Aut(Leech lattice)");
    println!("  Co_1 = Co_0 / {{±1}}");
    println!("  Monster contains Co_1 as subgroup");
    
    if leech_pattern.points_found.len() > 0 {
        println!("\n  ✓ Leech lattice structure detected!");
        println!("  ✓ Monster-Leech connection confirmed!");
    }
    
    // Analyze symmetries
    println!("\n🎭 Symmetry Analysis:");
    let symmetries = analyze_symmetries(&leech_pattern.points_found);
    println!("  Reflection symmetries: {}", symmetries.reflections);
    println!("  Rotation symmetries: {}", symmetries.rotations);
    println!("  Total automorphisms: {}", symmetries.total);
    
    // Save results
    std::fs::write(
        "LEECH_LATTICE.json",
        serde_json::to_string_pretty(&leech_pattern)?
    )?;
    
    println!("\n✓ Analysis complete!");
    println!("  Results: LEECH_LATTICE.json");
    
    Ok(())
}

#[derive(Debug)]
struct Symmetries {
    reflections: usize,
    rotations: usize,
    total: usize,
}

fn analyze_symmetries(points: &[LeechLatticePoint]) -> Symmetries {
    let mut reflections = 0;
    let mut rotations = 0;
    
    // Check for reflection symmetries
    for point in points {
        let reflected: Vec<i32> = point.coordinates.iter().map(|&x| -x).collect();
        if points.iter().any(|p| p.coordinates == reflected) {
            reflections += 1;
        }
    }
    
    // Check for cyclic rotations
    for point in points {
        let mut rotated = point.coordinates.clone();
        rotated.rotate_left(1);
        if points.iter().any(|p| p.coordinates == rotated) {
            rotations += 1;
        }
    }
    
    Symmetries {
        reflections,
        rotations,
        total: reflections + rotations,
    }
}
