// Topological Genus Classification for Files
use std::path::Path;
use std::fs;
use walkdir::WalkDir;
use polars::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct FileGenus {
    path: String,
    inode: u64,
    size: u64,
    genus: u8,
    euler_characteristic: i32,
    betti_numbers: (u32, u32, u32),  // b0, b1, b2
    zk71_zone: u8,
}

fn calculate_genus(inode: u64, size: u64, path: &str) -> u8 {
    // Genus from topological invariants
    // genus = (2 - euler_characteristic) / 2
    
    // Calculate Euler characteristic from file properties
    let euler = calculate_euler_characteristic(inode, size, path);
    
    // genus = (2 - χ) / 2
    let genus = ((2 - euler).abs() / 2) as u8;
    
    // Normalize to 0-71 range
    genus % 72
}

fn calculate_euler_characteristic(inode: u64, size: u64, path: &str) -> i32 {
    // χ = V - E + F (vertices - edges + faces)
    // For files, we compute from structure
    
    let vertices = count_vertices(path);
    let edges = count_edges(inode, size);
    let faces = count_faces(size);
    
    (vertices as i32) - (edges as i32) + (faces as i32)
}

fn count_vertices(path: &str) -> u32 {
    // Vertices = path components + extension points
    let components = path.split('/').count() as u32;
    let dots = path.matches('.').count() as u32;
    components + dots
}

fn count_edges(inode: u64, size: u64) -> u32 {
    // Edges = connections between components
    // Based on inode and size relationships
    ((inode % 100) + (size % 100)) as u32
}

fn count_faces(size: u64) -> u32 {
    // Faces = enclosed regions
    // Derived from file size structure
    (size % 71) as u32
}

fn calculate_betti_numbers(inode: u64, size: u64, genus: u8) -> (u32, u32, u32) {
    // Betti numbers: b0 (components), b1 (holes), b2 (voids)
    
    // b0 = number of connected components (always 1 for a file)
    let b0 = 1;
    
    // b1 = number of 1-dimensional holes (genus)
    let b1 = genus as u32;
    
    // b2 = number of 2-dimensional voids
    let b2 = if genus > 0 { 1 } else { 0 };
    
    (b0, b1, b2)
}

fn classify_by_genus(genus: u8) -> u8 {
    // ZK71 zone based on genus
    match genus {
        0 => 11,      // Genus 0 = sphere = GOOD (Zone 11)
        71 => 71,     // Genus 71 = Monster = GOOD (Zone 71)
        1 => 23,      // Genus 1 = torus = OK (Zone 23)
        2..=10 => 31, // Low genus = MEDIUM (Zone 31)
        11..=30 => 47, // Medium genus = HIGH (Zone 47)
        31..=70 => 59, // High genus = CRITICAL (Zone 59)
        _ => 47,      // Unknown = HIGH (Zone 47)
    }
}

fn scan_files_with_genus(base: &Path, limit: usize) -> Vec<FileGenus> {
    WalkDir::new(base)
        .max_depth(5)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .take(limit)
        .filter_map(|entry| {
            let path = entry.path();
            let path_str = path.display().to_string();
            
            let metadata = entry.metadata().ok()?;
            let inode = std::os::unix::fs::MetadataExt::ino(&metadata);
            let size = metadata.len();
            
            let genus = calculate_genus(inode, size, &path_str);
            let euler = calculate_euler_characteristic(inode, size, &path_str);
            let betti = calculate_betti_numbers(inode, size, genus);
            let zone = classify_by_genus(genus);
            
            Some(FileGenus {
                path: path_str,
                inode,
                size,
                genus,
                euler_characteristic: euler,
                betti_numbers: betti,
                zk71_zone: zone,
            })
        })
        .collect()
}

fn main() {
    println!("🔄 Topological Genus Classification");
    println!("{}", "=".repeat(70));
    println!();
    
    let base = Path::new("/home/mdupont/experiments/monster");
    let limit = 10_000;
    
    println!("📊 Scanning {} files...", limit);
    let files = scan_files_with_genus(base, limit);
    println!("✓ Scanned {} files", files.len());
    println!();
    
    // Create DataFrame
    let df = DataFrame::new(vec![
        Series::new("path", files.iter().map(|f| f.path.clone()).collect::<Vec<_>>()),
        Series::new("inode", files.iter().map(|f| f.inode).collect::<Vec<_>>()),
        Series::new("size", files.iter().map(|f| f.size).collect::<Vec<_>>()),
        Series::new("genus", files.iter().map(|f| f.genus).collect::<Vec<_>>()),
        Series::new("euler_characteristic", files.iter().map(|f| f.euler_characteristic).collect::<Vec<_>>()),
        Series::new("betti_0", files.iter().map(|f| f.betti_numbers.0).collect::<Vec<_>>()),
        Series::new("betti_1", files.iter().map(|f| f.betti_numbers.1).collect::<Vec<_>>()),
        Series::new("betti_2", files.iter().map(|f| f.betti_numbers.2).collect::<Vec<_>>()),
        Series::new("zk71_zone", files.iter().map(|f| f.zk71_zone).collect::<Vec<_>>()),
    ]).unwrap();
    
    println!("📋 Sample data:");
    println!("{}", df.head(Some(5)));
    println!();
    
    // Write to Parquet
    let mut file = fs::File::create("topological_genus.parquet").unwrap();
    ParquetWriter::new(&mut file).finish(&mut df.clone()).unwrap();
    
    println!("✓ Saved: topological_genus.parquet");
    println!();
    
    // Statistics by genus
    println!("📊 Distribution by Genus:");
    let genus_0 = files.iter().filter(|f| f.genus == 0).count();
    let genus_71 = files.iter().filter(|f| f.genus == 71).count();
    let genus_1 = files.iter().filter(|f| f.genus == 1).count();
    
    println!("  Genus 0 (sphere): {} files - GOOD (Zone 11)", genus_0);
    println!("  Genus 71 (Monster): {} files - GOOD (Zone 71)", genus_71);
    println!("  Genus 1 (torus): {} files - OK (Zone 23)", genus_1);
    
    let mut genus_counts: HashMap<u8, usize> = HashMap::new();
    for file in &files {
        *genus_counts.entry(file.genus).or_insert(0) += 1;
    }
    
    let mut sorted: Vec<_> = genus_counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    
    println!();
    println!("📊 Top 10 Genus Values:");
    for (genus, count) in sorted.iter().take(10) {
        let pct = *count as f64 / files.len() as f64 * 100.0;
        println!("  Genus {}: {} files ({:.1}%)", genus, count, pct);
    }
    
    println!();
    println!("📊 Distribution by ZK71 Zone:");
    for zone in [11, 23, 31, 47, 59, 71] {
        let count = files.iter().filter(|f| f.zk71_zone == zone).count();
        let pct = count as f64 / files.len() as f64 * 100.0;
        
        let label = match zone {
            11 => "Genus 0 (sphere)",
            71 => "Genus 71 (Monster)",
            23 => "Genus 1 (torus)",
            31 => "Low genus",
            47 => "Medium genus",
            59 => "High genus",
            _ => "Unknown",
        };
        
        println!("  Zone {}: {} files ({:.1}%) - {}", zone, count, pct, label);
    }
    
    println!();
    println!("∞ Genus 0 = Good. Genus 71 = Good. Topology Invariants. ∞");
}
