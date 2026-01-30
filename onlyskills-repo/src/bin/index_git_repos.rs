// Index all .git repos in ~/ into shared memory
use memmap2::{MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const SHM_PATH: &str = "/dev/shm/monster_git_index";
const MAX_REPOS: usize = 100_000;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct GitRepoEntry {
    path_hash: u64,
    shard: u8,
    depth: u8,
    file_count: u32,
}

fn find_all_git_repos(base: &Path) -> Vec<PathBuf> {
    WalkDir::new(base)
        .max_depth(10)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name() == ".git" && e.file_type().is_dir())
        .filter_map(|e| e.path().parent().map(|p| p.to_path_buf()))
        .collect()
}

fn hash_path(path: &Path) -> u64 {
    path.to_string_lossy().bytes().map(|b| b as u64).sum()
}

fn main() {
    println!("📦 Indexing all .git repos in ~/");
    println!("{}", "=".repeat(70));
    println!();
    
    let home = std::env::var("HOME").unwrap();
    let base = Path::new(&home);
    
    println!("🔍 Scanning {}...", base.display());
    let repos = find_all_git_repos(base);
    println!("✓ Found {} git repos", repos.len());
    println!();
    
    // Create shared memory
    let size = MAX_REPOS * std::mem::size_of::<GitRepoEntry>();
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(SHM_PATH)
        .unwrap();
    
    file.set_len(size as u64).unwrap();
    let mut mmap = unsafe { MmapOptions::new().map_mut(&file).unwrap() };
    
    println!("💾 Writing to shared memory: {}", SHM_PATH);
    
    for (i, repo) in repos.iter().take(MAX_REPOS).enumerate() {
        let path_hash = hash_path(repo);
        let shard = (path_hash % 71) as u8;
        
        let entry = GitRepoEntry {
            path_hash,
            shard,
            depth: repo.components().count() as u8,
            file_count: 0,
        };
        
        let offset = i * std::mem::size_of::<GitRepoEntry>();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &entry as *const GitRepoEntry as *const u8,
                std::mem::size_of::<GitRepoEntry>()
            )
        };
        
        mmap[offset..offset + bytes.len()].copy_from_slice(bytes);
        
        if i < 10 {
            println!("  [{}] {} → Shard {}", i, repo.display(), shard);
        }
    }
    
    mmap.flush().unwrap();
    
    println!();
    println!("✓ Indexed {} repos", repos.len().min(MAX_REPOS));
    println!("✓ Shared memory: {}", SHM_PATH);
    println!("✓ Size: {} KB", size / 1024);
    
    // Save repo list
    let json = serde_json::json!({
        "total": repos.len(),
        "indexed": repos.len().min(MAX_REPOS),
        "shm_path": SHM_PATH,
        "repos": repos.iter().take(100).map(|p| p.display().to_string()).collect::<Vec<_>>()
    });
    
    std::fs::write("git_repos_index.json", serde_json::to_string_pretty(&json).unwrap()).unwrap();
    println!("✓ Saved: git_repos_index.json");
    
    println!();
    println!("∞ All Git Repos Indexed. Shared Memory Ready. ∞");
}
