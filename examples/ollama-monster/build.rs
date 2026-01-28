// build.rs - Rewrite mistral.rs code to add Monster introspection

use std::env;
use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    
    // Check if mistral.rs source is available
    let mistral_path = env::var("MISTRAL_RS_PATH")
        .unwrap_or_else(|_| "../../../mistral.rs".to_string());
    
    if !Path::new(&mistral_path).exists() {
        println!("cargo:warning=mistral.rs not found at {}", mistral_path);
        println!("cargo:warning=Set MISTRAL_RS_PATH environment variable");
        return;
    }
    
    println!("cargo:warning=Found mistral.rs at {}", mistral_path);
    
    // Strategy: Patch mistral.rs source to add Monster introspection
    // 1. Find forward pass functions
    // 2. Inject Monster analysis calls
    // 3. Generate instrumented version
    
    let src_files = vec![
        "mistralrs-core/src/pipeline/mod.rs",
        "mistralrs-core/src/model.rs",
    ];
    
    for file in src_files {
        let full_path = format!("{}/{}", mistral_path, file);
        if let Ok(content) = fs::read_to_string(&full_path) {
            let instrumented = instrument_code(&content);
            
            // Save instrumented version
            let out_dir = env::var("OUT_DIR").unwrap();
            let out_file = format!("{}/instrumented_{}", out_dir, file.replace("/", "_"));
            fs::write(&out_file, instrumented).ok();
            
            println!("cargo:warning=Instrumented: {}", file);
        }
    }
}

fn instrument_code(source: &str) -> String {
    let mut output = String::new();
    
    // Add Monster imports at top
    output.push_str("use crate::monster_introspector::*;\n\n");
    output.push_str(source);
    
    // Find and instrument forward pass functions
    output = output.replace(
        "fn forward(",
        "#[monster_introspect]\nfn forward("
    );
    
    output = output.replace(
        "fn load_weights(",
        "#[trace_weights]\nfn load_weights("
    );
    
    // Inject analysis calls after tensor operations
    output = output.replace(
        "let output = matmul(",
        "let output = matmul(\nlet _primes = analyze_tensor!(output);"
    );
    
    output
}
