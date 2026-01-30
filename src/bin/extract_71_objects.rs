// Rust version of extract_71_objects.py
// Extract ALL mathematical objects with value 71 from LMFDB

use std::fs;
use std::path::Path;
use serde::{Serialize, Deserialize};
use syn::{visit::Visit, Expr, Lit};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Object71 {
    file: String,
    line: usize,
    name: String,
    value: i64,
    object_type: String,
}

struct Visitor71 {
    file_path: String,
    objects: Vec<Object71>,
}

impl<'ast> Visit<'ast> for Visitor71 {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        if let Expr::Lit(expr_lit) = expr {
            if let Lit::Int(lit_int) = &expr_lit.lit {
                if lit_int.base10_parse::<i64>().ok() == Some(71) {
                    self.objects.push(Object71 {
                        file: self.file_path.clone(),
                        line: lit_int.span().start().line,
                        name: "literal".to_string(),
                        value: 71,
                        object_type: "constant".to_string(),
                    });
                }
            }
        }
        syn::visit::visit_expr(self, expr);
    }
}

fn extract_from_rust_file(path: &Path) -> Vec<Object71> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    
    let syntax = match syn::parse_file(&content) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    
    let mut visitor = Visitor71 {
        file_path: path.display().to_string(),
        objects: Vec::new(),
    };
    
    visitor.visit_file(&syntax);
    visitor.objects
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 EXTRACTING ALL 71-VALUED OBJECTS");
    println!("{}", "=".repeat(70));
    println!();
    
    let lmfdb_path = "/mnt/data1/nix/source/github/meta-introspector/lmfdb";
    let mut all_objects = Vec::new();
    
    // Process Rust files
    for entry in walkdir::WalkDir::new(lmfdb_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
    {
        let objects = extract_from_rust_file(entry.path());
        all_objects.extend(objects);
    }
    
    println!("Found {} objects with value 71", all_objects.len());
    
    // Save to JSON
    let json = serde_json::to_string_pretty(&all_objects)?;
    fs::write("objects_71.json", json)?;
    
    println!("✅ Saved to objects_71.json");
    
    Ok(())
}
