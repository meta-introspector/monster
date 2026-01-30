// Trace mathematical software execution - Pure Rust

use std::process::Command;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
struct ExecutionTrace {
    software: String,
    example: String,
    complexity: f64,
    period: String,
    output: String,
    perf_data: HashMap<String, u64>,
}

struct MathTracer;

impl MathTracer {
    fn trace_gap(&self, example: &str) -> Result<ExecutionTrace, String> {
        println!("🔍 Tracing GAP: {}", example);
        
        let output = Command::new("nix-shell")
            .args(&["-p", "gap", "--run", &format!("gap -q -c '{}'", example)])
            .output()
            .map_err(|e| e.to_string())?;
        
        let result = String::from_utf8_lossy(&output.stdout).to_string();
        let complexity = self.calculate_complexity(&result);
        
        Ok(ExecutionTrace {
            software: "GAP".to_string(),
            example: example.to_string(),
            complexity,
            period: self.classify_period(complexity),
            output: result,
            perf_data: HashMap::new(),
        })
    }
    
    fn trace_pari(&self, example: &str) -> Result<ExecutionTrace, String> {
        println!("🔍 Tracing PARI/GP: {}", example);
        
        let output = Command::new("nix-shell")
            .args(&["-p", "pari", "--run", &format!("echo '{}' | gp -q", example)])
            .output()
            .map_err(|e| e.to_string())?;
        
        let result = String::from_utf8_lossy(&output.stdout).to_string();
        let complexity = self.calculate_complexity(&result);
        
        Ok(ExecutionTrace {
            software: "PARI".to_string(),
            example: example.to_string(),
            complexity,
            period: self.classify_period(complexity),
            output: result,
            perf_data: HashMap::new(),
        })
    }
    
    fn trace_sage(&self, example: &str) -> Result<ExecutionTrace, String> {
        println!("🔍 Tracing Sage: {}", example);
        
        let output = Command::new("nix-shell")
            .args(&["-p", "sage", "--run", &format!("sage -c '{}'", example)])
            .output()
            .map_err(|e| e.to_string())?;
        
        let result = String::from_utf8_lossy(&output.stdout).to_string();
        let complexity = self.calculate_complexity(&result);
        
        Ok(ExecutionTrace {
            software: "Sage".to_string(),
            example: example.to_string(),
            complexity,
            period: self.classify_period(complexity),
            output: result,
            perf_data: HashMap::new(),
        })
    }
    
    fn calculate_complexity(&self, output: &str) -> f64 {
        // Complexity based on output length and numeric values
        let base = output.len() as f64;
        
        // Extract numbers and use largest as complexity indicator
        let numbers: Vec<f64> = output
            .split_whitespace()
            .filter_map(|s| s.parse::<f64>().ok())
            .collect();
        
        let max_num = numbers.iter().max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(1.0);
        
        base * max_num.log10().max(1.0)
    }
    
    fn classify_period(&self, complexity: f64) -> String {
        match complexity {
            c if c < 10.0 => "Period1".to_string(),
            c if c < 100.0 => "Period2".to_string(),
            c if c < 1000.0 => "Period3".to_string(),
            c if c < 10000.0 => "Period4".to_string(),
            c if c < 100000.0 => "Period5".to_string(),
            c if c < 1000000.0 => "Period6".to_string(),
            c if c < 10000000.0 => "Period7".to_string(),
            c if c < 100000000.0 => "Period8".to_string(),
            c if c < 1000000000.0 => "Period9".to_string(),
            _ => "Period10".to_string(),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 MATHEMATICAL SOFTWARE TRACER");
    println!("{}", "=".repeat(70));
    println!();
    
    let tracer = MathTracer;
    let mut traces = Vec::new();
    
    // GAP examples
    println!("📐 GAP Examples:");
    let gap_examples = vec![
        "2+2;",
        "Order(SymmetricGroup(5));",
        "Order(AlternatingGroup(5));",
    ];
    
    for example in gap_examples {
        match tracer.trace_gap(example) {
            Ok(trace) => {
                println!("  ✅ {} → Period {}, complexity: {:.2}", 
                    example, trace.period, trace.complexity);
                traces.push(trace);
            }
            Err(e) => println!("  ⚠️  {}: {}", example, e),
        }
    }
    
    println!();
    println!("🔢 PARI/GP Examples:");
    let pari_examples = vec![
        "factor(12)",
        "ellap(ellinit([0,1]),7)",
        "fibonacci(20)",
    ];
    
    for example in pari_examples {
        match tracer.trace_pari(example) {
            Ok(trace) => {
                println!("  ✅ {} → Period {}, complexity: {:.2}", 
                    example, trace.period, trace.complexity);
                traces.push(trace);
            }
            Err(e) => println!("  ⚠️  {}: {}", example, e),
        }
    }
    
    println!();
    println!("🐍 Sage Examples:");
    let sage_examples = vec![
        "print(2+2)",
        "print(factor(100))",
        "print(EllipticCurve([0,1]).conductor())",
    ];
    
    for example in sage_examples {
        match tracer.trace_sage(example) {
            Ok(trace) => {
                println!("  ✅ {} → Period {}, complexity: {:.2}", 
                    example, trace.period, trace.complexity);
                traces.push(trace);
            }
            Err(e) => println!("  ⚠️  {}: {}", example, e),
        }
    }
    
    // Save traces
    println!();
    println!("💾 Saving traces...");
    std::fs::create_dir_all("analysis")?;
    
    let json = serde_json::to_string_pretty(&traces)?;
    std::fs::write("analysis/math_software_traces.json", json)?;
    
    println!("  ✅ analysis/math_software_traces.json");
    
    println!();
    println!("📊 Summary:");
    println!("  Total traces: {}", traces.len());
    
    let mut period_counts: HashMap<String, usize> = HashMap::new();
    for trace in &traces {
        *period_counts.entry(trace.period.clone()).or_default() += 1;
    }
    
    for (period, count) in period_counts {
        println!("  {}: {} traces", period, count);
    }
    
    println!();
    println!("✅ Mathematical software traced and classified!");
    
    Ok(())
}
