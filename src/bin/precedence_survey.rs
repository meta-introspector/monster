use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug, Serialize, Deserialize)]
struct PrecedenceRecord {
    system: String,
    project: String,
    file: String,
    line: usize,
    operator: String,
    precedence: u32,
    git_url: String,
    commit: String,
    branch: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Citation {
    system: String,
    project: String,
    git_url: String,
    commit: String,
    branch: String,
    scan_date: String,
    file_count: usize,
    precedence_count: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== COMPREHENSIVE PRECEDENCE SURVEY ===");
    
    let mut records = Vec::new();
    let mut citations = Vec::new();
    
    // 1. Scan Spectral (Lean2)
    if let Ok(spectral_data) = scan_spectral() {
        records.extend(spectral_data.0);
        citations.push(spectral_data.1);
    }
    
    // 2. Scan Lean4 Mathlib
    if let Ok(lean4_data) = scan_lean4() {
        records.extend(lean4_data.0);
        citations.push(lean4_data.1);
    }
    
    // 3. Scan Coq (if available)
    if let Ok(coq_data) = scan_coq() {
        records.extend(coq_data.0);
        citations.push(coq_data.1);
    }
    
    // 4. Clone and scan UniMath
    if let Ok(unimath_data) = scan_unimath() {
        records.extend(unimath_data.0);
        citations.push(unimath_data.1);
    }
    
    // 5. Clone and scan MetaCoq
    if let Ok(metacoq_data) = scan_metacoq() {
        records.extend(metacoq_data.0);
        citations.push(metacoq_data.1);
    }
    
    // Generate summary
    print_summary(&records, &citations);
    
    // Save to Parquet (via Nix store)
    save_to_parquet(&records, &citations)?;
    
    Ok(())
}

fn scan_spectral() -> Result<(Vec<PrecedenceRecord>, Citation), Box<dyn std::error::Error>> {
    println!("\n=== Scanning Spectral (Lean2) ===");
    
    let dir = PathBuf::from("spectral");
    if !dir.exists() {
        return Err("Spectral directory not found".into());
    }
    
    let (commit, branch) = get_git_info(&dir)?;
    
    let mut records = Vec::new();
    
    // Find all .hlean files
    for entry in walkdir::WalkDir::new(&dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "hlean"))
    {
        let content = fs::read_to_string(entry.path())?;
        
        for (line_num, line) in content.lines().enumerate() {
            if line.contains("infixl") || line.contains("infixr") {
                if let Some((operator, precedence)) = parse_lean_precedence(line) {
                    records.push(PrecedenceRecord {
                        system: "Lean2".to_string(),
                        project: "Spectral".to_string(),
                        file: entry.path().display().to_string(),
                        line: line_num + 1,
                        operator,
                        precedence,
                        git_url: "https://github.com/cmu-phil/Spectral".to_string(),
                        commit: commit.clone(),
                        branch: branch.clone(),
                    });
                }
            }
        }
    }
    
    let citation = Citation {
        system: "Lean2".to_string(),
        project: "Spectral".to_string(),
        git_url: "https://github.com/cmu-phil/Spectral".to_string(),
        commit,
        branch,
        scan_date: chrono::Utc::now().to_rfc3339(),
        file_count: walkdir::WalkDir::new(&dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "hlean"))
            .count(),
        precedence_count: records.len(),
    };
    
    println!("Found {} precedence declarations", records.len());
    
    Ok((records, citation))
}

fn scan_lean4() -> Result<(Vec<PrecedenceRecord>, Citation), Box<dyn std::error::Error>> {
    println!("\n=== Scanning Lean4 Mathlib ===");
    
    let dir = PathBuf::from(".lake/packages/mathlib");
    if !dir.exists() {
        return Err("Lean4 mathlib not found".into());
    }
    
    let (commit, branch) = get_git_info(&dir)?;
    
    let mut records = Vec::new();
    
    for entry in walkdir::WalkDir::new(&dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "lean"))
    {
        let content = fs::read_to_string(entry.path())?;
        
        for (line_num, line) in content.lines().enumerate() {
            if line.contains("infixl") || line.contains("infixr") {
                if let Some((operator, precedence)) = parse_lean_precedence(line) {
                    records.push(PrecedenceRecord {
                        system: "Lean4".to_string(),
                        project: "Mathlib".to_string(),
                        file: entry.path().display().to_string(),
                        line: line_num + 1,
                        operator,
                        precedence,
                        git_url: "https://github.com/leanprover-community/mathlib4".to_string(),
                        commit: commit.clone(),
                        branch: branch.clone(),
                    });
                }
            }
        }
    }
    
    let citation = Citation {
        system: "Lean4".to_string(),
        project: "Mathlib".to_string(),
        git_url: "https://github.com/leanprover-community/mathlib4".to_string(),
        commit,
        branch,
        scan_date: chrono::Utc::now().to_rfc3339(),
        file_count: walkdir::WalkDir::new(&dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "lean"))
            .count(),
        precedence_count: records.len(),
    };
    
    println!("Found {} precedence declarations", records.len());
    
    Ok((records, citation))
}

fn scan_coq() -> Result<(Vec<PrecedenceRecord>, Citation), Box<dyn std::error::Error>> {
    println!("\n=== Scanning Coq Stdlib ===");
    
    let dir = PathBuf::from(std::env::var("HOME")? + "/.opam/default/lib/coq");
    if !dir.exists() {
        return Err("Coq not found".into());
    }
    
    // Coq stdlib doesn't have git info in this location
    let commit = "system-install".to_string();
    let branch = "unknown".to_string();
    
    let mut records = Vec::new();
    
    for entry in walkdir::WalkDir::new(&dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "v"))
    {
        let content = fs::read_to_string(entry.path())?;
        
        for (line_num, line) in content.lines().enumerate() {
            if line.contains("at level") {
                if let Some((operator, precedence)) = parse_coq_precedence(line) {
                    records.push(PrecedenceRecord {
                        system: "Coq".to_string(),
                        project: "Stdlib".to_string(),
                        file: entry.path().display().to_string(),
                        line: line_num + 1,
                        operator,
                        precedence,
                        git_url: "https://github.com/coq/coq".to_string(),
                        commit: commit.clone(),
                        branch: branch.clone(),
                    });
                }
            }
        }
    }
    
    let citation = Citation {
        system: "Coq".to_string(),
        project: "Stdlib".to_string(),
        git_url: "https://github.com/coq/coq".to_string(),
        commit,
        branch,
        scan_date: chrono::Utc::now().to_rfc3339(),
        file_count: walkdir::WalkDir::new(&dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "v"))
            .count(),
        precedence_count: records.len(),
    };
    
    println!("Found {} precedence declarations", records.len());
    
    Ok((records, citation))
}

fn scan_unimath() -> Result<(Vec<PrecedenceRecord>, Citation), Box<dyn std::error::Error>> {
    println!("\n=== Scanning UniMath ===");
    
    let dir = PathBuf::from("external/UniMath");
    
    // Clone if not present
    if !dir.exists() {
        println!("Cloning UniMath...");
        Command::new("git")
            .args(&["clone", "--depth", "1", "https://github.com/UniMath/UniMath", "external/UniMath"])
            .status()?;
    }
    
    let (commit, branch) = get_git_info(&dir)?;
    
    let mut records = Vec::new();
    
    for entry in walkdir::WalkDir::new(&dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "v"))
    {
        let content = fs::read_to_string(entry.path())?;
        
        for (line_num, line) in content.lines().enumerate() {
            if line.contains("at level") {
                if let Some((operator, precedence)) = parse_coq_precedence(line) {
                    records.push(PrecedenceRecord {
                        system: "Coq".to_string(),
                        project: "UniMath".to_string(),
                        file: entry.path().display().to_string(),
                        line: line_num + 1,
                        operator,
                        precedence,
                        git_url: "https://github.com/UniMath/UniMath".to_string(),
                        commit: commit.clone(),
                        branch: branch.clone(),
                    });
                }
            }
        }
    }
    
    let citation = Citation {
        system: "Coq".to_string(),
        project: "UniMath".to_string(),
        git_url: "https://github.com/UniMath/UniMath".to_string(),
        commit,
        branch,
        scan_date: chrono::Utc::now().to_rfc3339(),
        file_count: walkdir::WalkDir::new(&dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "v"))
            .count(),
        precedence_count: records.len(),
    };
    
    println!("Found {} precedence declarations", records.len());
    
    Ok((records, citation))
}

fn scan_metacoq() -> Result<(Vec<PrecedenceRecord>, Citation), Box<dyn std::error::Error>> {
    println!("\n=== Scanning MetaCoq ===");
    
    let dir = PathBuf::from("external/metacoq");
    
    // Clone if not present
    if !dir.exists() {
        println!("Cloning MetaCoq...");
        Command::new("git")
            .args(&["clone", "--depth", "1", "https://github.com/MetaCoq/metacoq", "external/metacoq"])
            .status()?;
    }
    
    let (commit, branch) = get_git_info(&dir)?;
    
    let mut records = Vec::new();
    
    for entry in walkdir::WalkDir::new(&dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "v"))
    {
        let content = fs::read_to_string(entry.path())?;
        
        for (line_num, line) in content.lines().enumerate() {
            if line.contains("at level") {
                if let Some((operator, precedence)) = parse_coq_precedence(line) {
                    records.push(PrecedenceRecord {
                        system: "Coq".to_string(),
                        project: "MetaCoq".to_string(),
                        file: entry.path().display().to_string(),
                        line: line_num + 1,
                        operator,
                        precedence,
                        git_url: "https://github.com/MetaCoq/metacoq".to_string(),
                        commit: commit.clone(),
                        branch: branch.clone(),
                    });
                }
            }
        }
    }
    
    let citation = Citation {
        system: "Coq".to_string(),
        project: "MetaCoq".to_string(),
        git_url: "https://github.com/MetaCoq/metacoq".to_string(),
        commit,
        branch,
        scan_date: chrono::Utc::now().to_rfc3339(),
        file_count: walkdir::WalkDir::new(&dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "v"))
            .count(),
        precedence_count: records.len(),
    };
    
    println!("Found {} precedence declarations", records.len());
    
    Ok((records, citation))
}

fn get_git_info(dir: &PathBuf) -> Result<(String, String), Box<dyn std::error::Error>> {
    let commit = Command::new("git")
        .args(&["-C", dir.to_str().unwrap(), "rev-parse", "HEAD"])
        .output()?
        .stdout;
    let commit = String::from_utf8(commit)?.trim().to_string();
    
    let branch = Command::new("git")
        .args(&["-C", dir.to_str().unwrap(), "rev-parse", "--abbrev-ref", "HEAD"])
        .output()?
        .stdout;
    let branch = String::from_utf8(branch)?.trim().to_string();
    
    Ok((commit, branch))
}

fn parse_lean_precedence(line: &str) -> Option<(String, u32)> {
    // Parse: infixl ` ** `:71 := graded_ring.mul
    let re = regex::Regex::new(r#"infixl\s+`\s*([^`]+)`\s*:(\d+)"#).ok()?;
    let caps = re.captures(line)?;
    let operator = caps.get(1)?.as_str().trim().to_string();
    let precedence = caps.get(2)?.as_str().parse().ok()?;
    Some((operator, precedence))
}

fn parse_coq_precedence(line: &str) -> Option<(String, u32)> {
    // Parse: Notation "x + y" := (add x y) (at level 50, left associativity).
    let re = regex::Regex::new(r#"at level (\d+)"#).ok()?;
    let caps = re.captures(line)?;
    let precedence = caps.get(1)?.as_str().parse().ok()?;
    
    // Try to extract operator
    let op_re = regex::Regex::new(r#""([^"]+)""#).ok()?;
    let operator = op_re.captures(line)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    
    Some((operator, precedence))
}

fn print_summary(records: &[PrecedenceRecord], citations: &[Citation]) {
    println!("\n=== SUMMARY ===");
    println!("Total records: {}", records.len());
    println!("\nBy system:");
    
    let mut by_system = std::collections::HashMap::new();
    for record in records {
        *by_system.entry(&record.system).or_insert(0) += 1;
    }
    for (system, count) in by_system {
        println!("  {}: {}", system, count);
    }
    
    println!("\nPrecedence 71 occurrences: {}", 
        records.iter().filter(|r| r.precedence == 71).count());
    
    println!("\nFiles with precedence 71:");
    for record in records.iter().filter(|r| r.precedence == 71) {
        println!("  {} - {} - {} (line {})", 
            record.system, record.project, record.file, record.line);
    }
    
    println!("\nMonster prime distribution:");
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71] {
        let count = records.iter().filter(|r| r.precedence == p).count();
        println!("  Prime {}: {} occurrences", p, count);
    }
}

fn save_to_parquet(records: &[PrecedenceRecord], citations: &[Citation]) -> Result<(), Box<dyn std::error::Error>> {
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use std::fs::File;
    use std::sync::Arc;
    
    // Save records
    let records_json = serde_json::to_string(records)?;
    fs::write("datasets/precedence_survey/records.json", records_json)?;
    
    // Save citations
    let citations_json = serde_json::to_string(citations)?;
    fs::write("datasets/precedence_survey/citations.json", citations_json)?;
    
    println!("\n✓ Saved to datasets/precedence_survey/");
    
    Ok(())
}
