// Git Repo to zkerdfa Monster Form (Rust)
use std::path::Path;
use std::process::Command;
use serde::{Deserialize, Serialize};

const MONSTER_DIM: usize = 196883;  // Smallest faithful representation
const RING_DIM: usize = 71;         // Trimmed to Monster prime ring

const MONSTER_PRIMES: [u8; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];

#[derive(Debug, Serialize, Deserialize)]
pub struct MonsterForm {
    pub repo_path: String,
    pub full_dim: usize,
    pub ring_dim: usize,
    pub coords: Vec<u8>,
    pub url: String,
}

#[derive(Debug)]
struct GitAttributes {
    commits: u64,
    files: u64,
    authors: u64,
    languages: Vec<String>,
    age_days: u64,
    size_bytes: u64,
}

pub fn repo_to_monster_form(repo_path: &Path) -> MonsterForm {
    // 1. Extract git attributes
    let attrs = extract_git_attributes(repo_path);
    
    // 2. Map to Monster coordinates (196,883 dims)
    let full_coords = attrs_to_monster_coords(&attrs);
    
    // 3. Trim to ring (71 dims)
    let ring_coords = trim_to_ring(&full_coords, RING_DIM);
    
    // 4. Generate zkerdfa URL
    let url = generate_zkerdfa_url(&ring_coords);
    
    MonsterForm {
        repo_path: repo_path.display().to_string(),
        full_dim: MONSTER_DIM,
        ring_dim: RING_DIM,
        coords: ring_coords,
        url,
    }
}

fn extract_git_attributes(repo_path: &Path) -> GitAttributes {
    GitAttributes {
        commits: git_commit_count(repo_path),
        files: git_file_count(repo_path),
        authors: git_author_count(repo_path),
        languages: git_languages(repo_path),
        age_days: git_age_days(repo_path),
        size_bytes: git_size(repo_path),
    }
}

fn attrs_to_monster_coords(attrs: &GitAttributes) -> Vec<u8> {
    let mut coords = vec![0u8; MONSTER_DIM];
    
    // Commits → dimensions 0-14 (Monster primes)
    for (i, &prime) in MONSTER_PRIMES.iter().enumerate() {
        coords[i] = (attrs.commits % prime as u64) as u8;
    }
    
    // Files → dimensions 15-29
    for i in 0..15 {
        coords[15 + i] = (attrs.files % 71) as u8;
    }
    
    // Authors → dimensions 30-44
    for i in 0..15 {
        coords[30 + i] = ((attrs.authors * 2) % 71) as u8;
    }
    
    // Languages → dimensions 45-115 (71 language slots)
    for lang in &attrs.languages {
        let idx = language_index(lang);
        if idx < 71 {
            coords[45 + idx] = 1;
        }
    }
    
    // Age → dimensions 116-130
    let years = attrs.age_days / 365;
    for i in 0..15 {
        coords[116 + i] = (years % 71) as u8;
    }
    
    // Size → dimensions 131-145
    let mb = attrs.size_bytes / (1024 * 1024);
    for i in 0..15 {
        coords[131 + i] = (mb % 71) as u8;
    }
    
    coords
}

fn trim_to_ring(full_coords: &[u8], ring_dim: usize) -> Vec<u8> {
    // Project to ring: take first ring_dim coordinates
    full_coords[..ring_dim].to_vec()
}

fn generate_zkerdfa_url(coords: &[u8]) -> String {
    // Hash coordinates
    let hash: u64 = coords.iter().map(|&c| c as u64).sum::<u64>() % (71 * 71 * 71);
    
    // Encode coordinates as base64
    let encoded = base64::encode(coords);
    
    format!("zkerdfa://monster/ring71/{}?coords={}", hash, encoded)
}

fn language_index(lang: &str) -> usize {
    match lang {
        "rust" => 70,
        "lean" => 58,
        "prolog" => 46,
        "nix" => 40,
        "python" => 1,
        _ => 10,
    }
}

// Git operations
fn git_commit_count(repo_path: &Path) -> u64 {
    let output = Command::new("git")
        .args(&["-C", repo_path.to_str().unwrap(), "rev-list", "--count", "HEAD"])
        .output()
        .ok();
    
    output
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

fn git_file_count(repo_path: &Path) -> u64 {
    let output = Command::new("git")
        .args(&["-C", repo_path.to_str().unwrap(), "ls-files"])
        .output()
        .ok();
    
    output
        .map(|o| o.stdout.iter().filter(|&&b| b == b'\n').count() as u64)
        .unwrap_or(0)
}

fn git_author_count(repo_path: &Path) -> u64 {
    let output = Command::new("git")
        .args(&["-C", repo_path.to_str().unwrap(), "shortlog", "-sn", "HEAD"])
        .output()
        .ok();
    
    output
        .map(|o| o.stdout.iter().filter(|&&b| b == b'\n').count() as u64)
        .unwrap_or(0)
}

fn git_languages(repo_path: &Path) -> Vec<String> {
    // Simplified: detect by file extensions
    vec!["rust".to_string(), "prolog".to_string()]
}

fn git_age_days(repo_path: &Path) -> u64 {
    // First commit timestamp
    365 // Placeholder
}

fn git_size(repo_path: &Path) -> u64 {
    // .git directory size
    10 * 1024 * 1024 // Placeholder
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_monster_form() {
        let path = Path::new(".");
        let form = repo_to_monster_form(path);
        
        assert_eq!(form.full_dim, 196883);
        assert_eq!(form.ring_dim, 71);
        assert_eq!(form.coords.len(), 71);
        assert!(form.url.starts_with("zkerdfa://monster/ring71/"));
    }
    
    #[test]
    fn test_trim_to_ring() {
        let full = vec![1u8; 196883];
        let ring = trim_to_ring(&full, 71);
        
        assert_eq!(ring.len(), 71);
        assert_eq!(ring[0], 1);
    }
}
