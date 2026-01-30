// 71-Shard Full Spectrum Inode Analysis
use polars::prelude::*;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use walkdir::WalkDir;

#[derive(Debug, Clone)]
struct InodeRecord {
    inode: u64,
    path: String,
    size: u64,
    blocks: u64,
    nlink: u64,
    mode: u32,
    uid: u32,
    gid: u32,
    atime: i64,
    mtime: i64,
    ctime: i64,
    // Shard assignments by different attributes
    shard_by_inode: u8,
    shard_by_size: u8,
    shard_by_path: u8,
    shard_by_time: u8,
    shard_by_owner: u8,
    shard_by_selinux: u8,
    shard_by_git_project: u8,
    shard_by_language: u8,
    shard_by_last_used: u8,
    shard_by_file_size: u8,
    // Prime harmonics (15 Monster primes)
    harmonic_2: f64,
    harmonic_3: f64,
    harmonic_5: f64,
    harmonic_7: f64,
    harmonic_11: f64,
    harmonic_13: f64,
    harmonic_17: f64,
    harmonic_19: f64,
    harmonic_23: f64,
    harmonic_29: f64,
    harmonic_31: f64,
    harmonic_41: f64,
    harmonic_47: f64,
    harmonic_59: f64,
    harmonic_71: f64,
    // Frequency in RRDB (round-robin database)
    rrdb_frequency: f64,
}

fn shard_by_attribute(value: u64, attr: &str) -> u8 {
    (value % 71) as u8
}

const MONSTER_PRIMES: [u64; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];

fn calculate_harmonics(value: u64) -> [f64; 15] {
    let mut harmonics = [0.0; 15];
    for (i, &prime) in MONSTER_PRIMES.iter().enumerate() {
        // Harmonic = sin(2π * value / prime)
        harmonics[i] = ((2.0 * std::f64::consts::PI * value as f64) / prime as f64).sin();
    }
    harmonics
}

fn calculate_rrdb_frequency(atime: i64, mtime: i64, ctime: i64) -> f64 {
    // RRDB frequency = accesses per day
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    
    let age_days = (now - ctime) as f64 / 86400.0;
    let last_access_days = (now - atime) as f64 / 86400.0;
    
    if age_days > 0.0 && last_access_days < age_days {
        1.0 / last_access_days  // Higher frequency = more recent access
    } else {
        0.0
    }
}

fn scan_inodes(base: &Path, limit: usize) -> Vec<InodeRecord> {
    WalkDir::new(base)
        .max_depth(5)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .take(limit)
        .filter_map(|entry| {
            let metadata = entry.metadata().ok()?;
            let path = entry.path().display().to_string();
            
            let inode = metadata.ino();
            let size = metadata.size();
            let mtime = metadata.mtime();
            let atime = metadata.atime();
            let ctime = metadata.ctime();
            let uid = metadata.uid();
            
            // Get SELinux context
            let selinux_zone = get_selinux_zone(entry.path());
            
            // Get git project
            let git_project = find_git_project(entry.path());
            
            // Detect language
            let language = detect_language(entry.path());
            
            // Calculate harmonics
            let harmonics = calculate_harmonics(inode);
            
            // Calculate RRDB frequency
            let rrdb_freq = calculate_rrdb_frequency(atime, mtime, ctime);
            
            Some(InodeRecord {
                inode,
                path: path.clone(),
                size,
                blocks: metadata.blocks(),
                nlink: metadata.nlink(),
                mode: metadata.mode(),
                uid,
                gid: metadata.gid(),
                atime,
                mtime,
                ctime,
                shard_by_inode: shard_by_attribute(inode, "inode"),
                shard_by_size: shard_by_attribute(size, "size"),
                shard_by_path: shard_by_attribute(path.bytes().map(|b| b as u64).sum(), "path"),
                shard_by_time: shard_by_attribute(mtime as u64, "time"),
                shard_by_owner: shard_by_attribute(uid as u64, "owner"),
                shard_by_selinux: selinux_zone,
                shard_by_git_project: git_project,
                shard_by_language: language,
                shard_by_last_used: shard_by_attribute(atime as u64, "last_used"),
                shard_by_file_size: shard_by_attribute(size, "file_size"),
                harmonic_2: harmonics[0],
                harmonic_3: harmonics[1],
                harmonic_5: harmonics[2],
                harmonic_7: harmonics[3],
                harmonic_11: harmonics[4],
                harmonic_13: harmonics[5],
                harmonic_17: harmonics[6],
                harmonic_19: harmonics[7],
                harmonic_23: harmonics[8],
                harmonic_29: harmonics[9],
                harmonic_31: harmonics[10],
                harmonic_41: harmonics[11],
                harmonic_47: harmonics[12],
                harmonic_59: harmonics[13],
                harmonic_71: harmonics[14],
                rrdb_frequency: rrdb_freq,
            })
        })
        .collect()
}

fn get_selinux_zone(path: &Path) -> u8 {
    // Read SELinux context and map to zone
    // ls -Z would show: user:role:type:level
    let path_str = path.to_string_lossy();
    
    if path_str.contains("vile_code") { 71 }
    else if path_str.contains("quarantine") { 59 }
    else if path_str.contains("untrusted") { 47 }
    else if path_str.contains("tmp") { 31 }
    else { 11 }
}

fn find_git_project(path: &Path) -> u8 {
    // Walk up to find .git directory
    let mut current = path;
    
    while let Some(parent) = current.parent() {
        if parent.join(".git").exists() {
            let project_name = parent.file_name()
                .unwrap_or_default()
                .to_string_lossy();
            return (project_name.bytes().map(|b| b as u64).sum() % 71) as u8;
        }
        current = parent;
    }
    
    0  // No git project
}

fn detect_language(path: &Path) -> u8 {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    
    match ext {
        "rs" => 71,   // Rust = highest
        "lean" => 59, // Lean4
        "pl" => 47,   // Prolog
        "nix" => 41,  // Nix
        "mzn" => 31,  // MiniZinc
        "c" | "cpp" => 23,
        "js" | "ts" => 17,
        "py" => 2,    // Python = lowest (forbidden)
        _ => 11,
    }
}

fn main() {
    println!("🔀 71-Shard Full Spectrum Inode Analysis");
    println!("{}", "=".repeat(70));
    println!();
    
    let base = Path::new("/home/mdupont/experiments/monster");
    let limit = 10_000;
    
    println!("📊 Scanning {} files...", limit);
    let records = scan_inodes(base, limit);
    println!("✓ Scanned {} inodes", records.len());
    println!();
    
    // Create DataFrame with all shard assignments
    let df = DataFrame::new(vec![
        Series::new("inode", records.iter().map(|r| r.inode).collect::<Vec<_>>()),
        Series::new("path", records.iter().map(|r| r.path.clone()).collect::<Vec<_>>()),
        Series::new("size", records.iter().map(|r| r.size).collect::<Vec<_>>()),
        Series::new("atime", records.iter().map(|r| r.atime).collect::<Vec<_>>()),
        Series::new("mtime", records.iter().map(|r| r.mtime).collect::<Vec<_>>()),
        Series::new("ctime", records.iter().map(|r| r.ctime).collect::<Vec<_>>()),
        Series::new("shard_by_inode", records.iter().map(|r| r.shard_by_inode).collect::<Vec<_>>()),
        Series::new("shard_by_size", records.iter().map(|r| r.shard_by_size).collect::<Vec<_>>()),
        Series::new("shard_by_path", records.iter().map(|r| r.shard_by_path).collect::<Vec<_>>()),
        Series::new("shard_by_time", records.iter().map(|r| r.shard_by_time).collect::<Vec<_>>()),
        Series::new("shard_by_owner", records.iter().map(|r| r.shard_by_owner).collect::<Vec<_>>()),
        Series::new("shard_by_selinux", records.iter().map(|r| r.shard_by_selinux).collect::<Vec<_>>()),
        Series::new("shard_by_git_project", records.iter().map(|r| r.shard_by_git_project).collect::<Vec<_>>()),
        Series::new("shard_by_language", records.iter().map(|r| r.shard_by_language).collect::<Vec<_>>()),
        Series::new("shard_by_last_used", records.iter().map(|r| r.shard_by_last_used).collect::<Vec<_>>()),
        Series::new("shard_by_file_size", records.iter().map(|r| r.shard_by_file_size).collect::<Vec<_>>()),
        Series::new("harmonic_2", records.iter().map(|r| r.harmonic_2).collect::<Vec<_>>()),
        Series::new("harmonic_3", records.iter().map(|r| r.harmonic_3).collect::<Vec<_>>()),
        Series::new("harmonic_5", records.iter().map(|r| r.harmonic_5).collect::<Vec<_>>()),
        Series::new("harmonic_7", records.iter().map(|r| r.harmonic_7).collect::<Vec<_>>()),
        Series::new("harmonic_11", records.iter().map(|r| r.harmonic_11).collect::<Vec<_>>()),
        Series::new("harmonic_13", records.iter().map(|r| r.harmonic_13).collect::<Vec<_>>()),
        Series::new("harmonic_17", records.iter().map(|r| r.harmonic_17).collect::<Vec<_>>()),
        Series::new("harmonic_19", records.iter().map(|r| r.harmonic_19).collect::<Vec<_>>()),
        Series::new("harmonic_23", records.iter().map(|r| r.harmonic_23).collect::<Vec<_>>()),
        Series::new("harmonic_29", records.iter().map(|r| r.harmonic_29).collect::<Vec<_>>()),
        Series::new("harmonic_31", records.iter().map(|r| r.harmonic_31).collect::<Vec<_>>()),
        Series::new("harmonic_41", records.iter().map(|r| r.harmonic_41).collect::<Vec<_>>()),
        Series::new("harmonic_47", records.iter().map(|r| r.harmonic_47).collect::<Vec<_>>()),
        Series::new("harmonic_59", records.iter().map(|r| r.harmonic_59).collect::<Vec<_>>()),
        Series::new("harmonic_71", records.iter().map(|r| r.harmonic_71).collect::<Vec<_>>()),
        Series::new("rrdb_frequency", records.iter().map(|r| r.rrdb_frequency).collect::<Vec<_>>()),
    ]).unwrap();
    
    println!("📋 Sample data:");
    println!("{}", df.head(Some(5)));
    println!();
    
    // Write to Parquet
    let mut file = fs::File::create("inode_71_shards.parquet").unwrap();
    ParquetWriter::new(&mut file).finish(&mut df.clone()).unwrap();
    
    println!("✓ Saved: inode_71_shards.parquet");
    println!();
    
    // Statistics per shard attribute
    println!("📊 Shard Distribution:");
    for attr in ["inode", "size", "path", "time", "owner", "selinux", "git_project", "language", "last_used", "file_size"] {
        let col = format!("shard_by_{}", attr);
        if let Ok(series) = df.column(&col) {
            let unique = series.n_unique().unwrap();
            println!("  By {}: {} unique shards", attr, unique);
        }
    }
    
    println!();
    println!("🎵 Prime Harmonics:");
    for prime in MONSTER_PRIMES {
        let col = format!("harmonic_{}", prime);
        if let Ok(series) = df.column(&col) {
            let mean = series.mean().unwrap_or(0.0);
            println!("  Prime {}: mean = {:.4}", prime, mean);
        }
    }
    
    println!();
    println!("📈 RRDB Frequency:");
    if let Ok(series) = df.column("rrdb_frequency") {
        let mean = series.mean().unwrap_or(0.0);
        let max = series.max::<f64>().unwrap_or(0.0);
        println!("  Mean: {:.4} accesses/day", mean);
        println!("  Max: {:.4} accesses/day", max);
    }
    
    println!();
    println!("∞ 10 Shards. 15 Harmonics. RRDB Frequency. Full Spectrum. ∞");
}
