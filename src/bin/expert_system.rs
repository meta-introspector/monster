// Expert System: Merge Monster + zkPrologML

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::process::Command;

#[derive(Debug, Serialize, Deserialize)]
struct CodeMapping {
    keyword: String,
    group: u8,
    area: String,
    files_found: Vec<FileMatch>,
}

#[derive(Debug, Serialize, Deserialize)]
struct FileMatch {
    path: String,
    language: String,
    line_count: usize,
}

struct ExpertSystem {
    zkprologml_path: String,
    monster_path: String,
}

impl ExpertSystem {
    fn new() -> Self {
        Self {
            zkprologml_path: "/mnt/data1/nix/vendor/rust/github".to_string(),
            monster_path: "/home/mdupont/experiments/monster".to_string(),
        }
    }
    
    fn search_keyword(&self, keyword: &str) -> Vec<FileMatch> {
        let mut matches = Vec::new();
        
        // Search in both codebases
        for (name, path) in [
            ("zkPrologML", &self.zkprologml_path),
            ("Monster", &self.monster_path),
        ] {
            println!("  Searching {} for '{}'...", name, keyword);
            
            // Search each language
            for (lang, ext) in [
                ("Nix", "nix"),
                ("Rust", "rs"),
                ("Lean4", "lean"),
                ("Prolog", "pl"),
                ("MiniZinc", "mzn"),
                ("Coq", "v"),
            ] {
                if let Ok(files) = self.search_files(path, keyword, ext) {
                    for file in files {
                        matches.push(FileMatch {
                            path: file,
                            language: lang.to_string(),
                            line_count: 0, // TODO: count lines
                        });
                    }
                }
            }
        }
        
        matches
    }
    
    fn search_files(&self, base_path: &str, keyword: &str, ext: &str) -> Result<Vec<String>, String> {
        let output = Command::new("find")
            .args(&[
                base_path,
                "-name",
                &format!("*.{}", ext),
                "-type",
                "f",
                "-exec",
                "grep",
                "-l",
                "-i",
                keyword,
                "{}",
                ";"
            ])
            .output()
            .map_err(|e| e.to_string())?;
        
        let files = String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(|s| s.to_string())
            .collect();
        
        Ok(files)
    }
    
    fn build_code_map(&self, keywords: &[(u8, String, String)]) -> Vec<CodeMapping> {
        let mut mappings = Vec::new();
        
        for (group, area, keyword) in keywords {
            println!("🔍 Mapping keyword: {} (Group {}: {})", keyword, group, area);
            
            let files = self.search_keyword(keyword);
            
            if !files.is_empty() {
                println!("  ✅ Found {} files", files.len());
                
                mappings.push(CodeMapping {
                    keyword: keyword.clone(),
                    group: *group,
                    area: area.clone(),
                    files_found: files,
                });
            } else {
                println!("  ⚠️  No files found");
            }
        }
        
        mappings
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 EXPERT SYSTEM: Monster ⊕ zkPrologML");
    println!("{}", "=".repeat(70));
    println!();
    
    let expert = ExpertSystem::new();
    
    // Sample keywords from each group (5 per group for demo)
    let sample_keywords = vec![
        // Group 1: K-theory
        (1, "Complex K-theory".to_string(), "k-theory".to_string()),
        (1, "Complex K-theory".to_string(), "bott".to_string()),
        (1, "Complex K-theory".to_string(), "periodicity".to_string()),
        (1, "Complex K-theory".to_string(), "clifford".to_string()),
        (1, "Complex K-theory".to_string(), "spectrum".to_string()),
        
        // Group 2: Elliptic curves
        (2, "Elliptic curves".to_string(), "elliptic".to_string()),
        (2, "Elliptic curves".to_string(), "curve".to_string()),
        (2, "Elliptic curves".to_string(), "weierstrass".to_string()),
        (2, "Elliptic curves".to_string(), "j_invariant".to_string()),
        (2, "Elliptic curves".to_string(), "modular".to_string()),
        
        // Group 6: Monster moonshine
        (6, "Monster moonshine".to_string(), "monster".to_string()),
        (6, "Monster moonshine".to_string(), "moonshine".to_string()),
        (6, "Monster moonshine".to_string(), "monstrous".to_string()),
        (6, "Monster moonshine".to_string(), "vertex".to_string()),
        (6, "Monster moonshine".to_string(), "mckay".to_string()),
    ];
    
    println!("📊 Building code map for {} sample keywords...", sample_keywords.len());
    println!();
    
    let mappings = expert.build_code_map(&sample_keywords);
    
    // Save results
    std::fs::create_dir_all("analysis/code_map")?;
    
    let json = serde_json::to_string_pretty(&mappings)?;
    std::fs::write("analysis/code_map/sample_mapping.json", json)?;
    
    println!();
    println!("📈 Summary:");
    println!("  Total keywords searched: {}", sample_keywords.len());
    println!("  Keywords with matches: {}", mappings.len());
    println!("  Total files found: {}", 
        mappings.iter().map(|m| m.files_found.len()).sum::<usize>());
    
    // Group by language
    let mut by_lang: HashMap<String, usize> = HashMap::new();
    for mapping in &mappings {
        for file in &mapping.files_found {
            *by_lang.entry(file.language.clone()).or_default() += 1;
        }
    }
    
    println!();
    println!("  By language:");
    for (lang, count) in by_lang {
        println!("    {}: {} files", lang, count);
    }
    
    println!();
    println!("💾 Saved to: analysis/code_map/sample_mapping.json");
    println!();
    println!("✅ Expert system complete!");
    println!();
    println!("🎯 Next: Run with all 710 keywords to build complete code map");
    
    Ok(())
}
