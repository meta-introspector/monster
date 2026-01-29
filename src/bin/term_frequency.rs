use std::collections::HashMap;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;
use regex::Regex;

fn main() {
    let extensions = vec!["rs", "lean", "md", "tex"];
    
    for ext in extensions {
        println!("\n## Files: *.{}\n", ext);
        
        let mut term_counts: HashMap<String, usize> = HashMap::new();
        
        let terms = vec![
            "Monster", "group", "prime", "Hecke", "modular", "eigenform",
            "proof", "theorem", "lemma", "conjecture",
            "digit", "factor", "hierarchical", "preservation",
            "Lean4", "Rust", "Python", "bisimulation",
        ];
        
        let patterns: Vec<_> = terms.iter()
            .map(|t| (t.to_string(), Regex::new(&format!(r"(?i)\b{}\b", t)).unwrap()))
            .collect();
        
        for entry in WalkDir::new(".")
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_type().is_file() && 
                e.path().extension().map_or(false, |e| e == ext) &&
                !e.path().to_str().unwrap().contains("target/")
            })
        {
            if let Ok(content) = fs::read_to_string(entry.path()) {
                for (term, pattern) in &patterns {
                    let count = pattern.find_iter(&content).count();
                    *term_counts.entry(term.clone()).or_insert(0) += count;
                }
            }
        }
        
        let mut sorted: Vec<_> = term_counts.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        
        println!("| Term | Count |");
        println!"|------|-------|");
        for (term, count) in sorted.iter().take(10) {
            if **count > 0 {
                println!("| {} | {} |", term, count);
            }
        }
    }
}
