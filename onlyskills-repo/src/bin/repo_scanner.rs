// Repo Scanner - Rust (replaces scan_all_repos.py)
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize)]
struct RepoScan {
    repo: PathBuf,
    name: String,
    shard: u8,
    doc_count: usize,
    docs: Vec<PathBuf>,
}

fn hash_to_shard(data: &str) -> u8 {
    (data.bytes().map(|b| b as u32).sum::<u32>() % 71) as u8
}

fn find_all_repos(base: &Path) -> Vec<PathBuf> {
    WalkDir::new(base)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name() == ".git" && e.file_type().is_dir())
        .filter_map(|e| e.path().parent().map(|p| p.to_path_buf()))
        .collect()
}

fn find_docs(repo: &Path) -> Vec<PathBuf> {
    WalkDir::new(repo)
        .max_depth(3)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().is_file() && 
            e.path().extension().map_or(false, |ext| 
                ext == "md" || ext == "txt" || ext == "rst"
            )
        })
        .map(|e| e.path().to_path_buf())
        .collect()
}

fn scan_repo(repo: &Path) -> RepoScan {
    let name = repo.file_name().unwrap().to_string_lossy().to_string();
    let docs = find_docs(repo);
    let shard = hash_to_shard(&name);
    
    RepoScan {
        repo: repo.to_path_buf(),
        name,
        shard,
        doc_count: docs.len(),
        docs: docs.into_iter().take(10).collect(),
    }
}

fn main() {
    println!("🔍 Repo Scanner (Rust)");
    println!("{}", "=".repeat(70));
    
    let base = Path::new("/home/mdupont");
    let repos = find_all_repos(base);
    
    println!("Found {} repos", repos.len());
    
    let scanned: Vec<_> = repos.iter().take(100)
        .map(|r| scan_repo(r))
        .collect();
    
    let mut by_shard: HashMap<u8, Vec<&RepoScan>> = HashMap::new();
    for scan in &scanned {
        by_shard.entry(scan.shard).or_default().push(scan);
    }
    
    println!("Distributed across {} shards", by_shard.len());
    
    let json = serde_json::to_string_pretty(&scanned).unwrap();
    std::fs::write("repo_scan_rust.json", json).unwrap();
    
    println!("✓ Saved: repo_scan_rust.json");
}
