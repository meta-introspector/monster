// SELinux Zone Assignment for Each File
use std::process::Command;
use std::path::Path;
use walkdir::WalkDir;
use polars::prelude::*;
use std::fs::File;

#[derive(Debug, Clone)]
struct FileZone {
    path: String,
    inode: u64,
    selinux_context: String,
    selinux_type: String,
    selinux_level: String,
    zk71_zone: u8,
}

fn get_selinux_context(path: &Path) -> Option<String> {
    let output = Command::new("ls")
        .arg("-Z")
        .arg(path)
        .output()
        .ok()?;
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Format: user:role:type:level path
    stdout.split_whitespace().next().map(|s| s.to_string())
}

fn parse_selinux_context(context: &str) -> (String, String) {
    let parts: Vec<&str> = context.split(':').collect();
    
    let selinux_type = parts.get(2).unwrap_or(&"unknown").to_string();
    let selinux_level = parts.get(3).unwrap_or(&"s0").to_string();
    
    (selinux_type, selinux_level)
}

fn assign_zk71_zone(selinux_type: &str, path: &str) -> u8 {
    // Map SELinux types to ZK71 zones
    match selinux_type {
        // Zone 71: CATASTROPHIC (vile code)
        t if t.contains("vile") => 71,
        t if t.contains("malicious") => 71,
        
        // Zone 59: CRITICAL (quarantine)
        t if t.contains("quarantine") => 59,
        t if t.contains("untrusted") => 59,
        
        // Zone 47: HIGH (suspicious)
        t if t.contains("tmp") => 47,
        t if t.contains("var_tmp") => 47,
        
        // Zone 31: MEDIUM (user content)
        t if t.contains("user") => 31,
        t if t.contains("home") => 31,
        
        // Zone 23: LOW-MEDIUM (system)
        t if t.contains("system") => 23,
        
        // Zone 11: LOW (safe)
        t if t.contains("lib") => 11,
        t if t.contains("bin") => 11,
        
        // Zone 2: MINIMAL (Python - forbidden)
        t if path.ends_with(".py") => 2,
        
        // Default: Zone 11
        _ => 11,
    }
}

fn scan_files_with_selinux(base: &Path, limit: usize) -> Vec<FileZone> {
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
            
            // Get SELinux context
            let context = get_selinux_context(path)
                .unwrap_or_else(|| "unknown:unknown:unknown:s0".to_string());
            
            let (selinux_type, selinux_level) = parse_selinux_context(&context);
            let zk71_zone = assign_zk71_zone(&selinux_type, &path_str);
            
            Some(FileZone {
                path: path_str,
                inode,
                selinux_context: context,
                selinux_type,
                selinux_level,
                zk71_zone,
            })
        })
        .collect()
}

fn main() {
    println!("🔐 SELinux Zone Assignment");
    println!("{}", "=".repeat(70));
    println!();
    
    let base = Path::new("/home/mdupont/experiments/monster");
    let limit = 10_000;
    
    println!("📊 Scanning {} files...", limit);
    let zones = scan_files_with_selinux(base, limit);
    println!("✓ Scanned {} files", zones.len());
    println!();
    
    // Create DataFrame
    let df = DataFrame::new(vec![
        Series::new("path", zones.iter().map(|z| z.path.clone()).collect::<Vec<_>>()),
        Series::new("inode", zones.iter().map(|z| z.inode).collect::<Vec<_>>()),
        Series::new("selinux_context", zones.iter().map(|z| z.selinux_context.clone()).collect::<Vec<_>>()),
        Series::new("selinux_type", zones.iter().map(|z| z.selinux_type.clone()).collect::<Vec<_>>()),
        Series::new("selinux_level", zones.iter().map(|z| z.selinux_level.clone()).collect::<Vec<_>>()),
        Series::new("zk71_zone", zones.iter().map(|z| z.zk71_zone).collect::<Vec<_>>()),
    ]).unwrap();
    
    println!("📋 Sample data:");
    println!("{}", df.head(Some(5)));
    println!();
    
    // Write to Parquet
    let mut file = File::create("selinux_zones.parquet").unwrap();
    ParquetWriter::new(&mut file).finish(&mut df.clone()).unwrap();
    
    println!("✓ Saved: selinux_zones.parquet");
    println!();
    
    // Statistics by zone
    println!("📊 Distribution by ZK71 Zone:");
    for zone in [2, 11, 23, 31, 47, 59, 71] {
        let count = zones.iter().filter(|z| z.zk71_zone == zone).count();
        let pct = count as f64 / zones.len() as f64 * 100.0;
        
        let label = match zone {
            71 => "CATASTROPHIC (vile)",
            59 => "CRITICAL (quarantine)",
            47 => "HIGH (suspicious)",
            31 => "MEDIUM (user)",
            23 => "LOW-MEDIUM (system)",
            11 => "LOW (safe)",
            2 => "MINIMAL (Python)",
            _ => "UNKNOWN",
        };
        
        println!("  Zone {}: {} files ({:.1}%) - {}", zone, count, pct, label);
    }
    
    println!();
    
    // Statistics by SELinux type
    println!("📊 Top SELinux Types:");
    let mut type_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for zone in &zones {
        *type_counts.entry(zone.selinux_type.clone()).or_insert(0) += 1;
    }
    
    let mut types: Vec<_> = type_counts.into_iter().collect();
    types.sort_by(|a, b| b.1.cmp(&a.1));
    
    for (selinux_type, count) in types.iter().take(10) {
        println!("  {}: {} files", selinux_type, count);
    }
    
    println!();
    println!("∞ SELinux Zones. ZK71 Security. Every File Classified. ∞");
}
