// Rust analyzer for process intervention
use std::fs;
use std::process::Command;

const LIFE_NUMBER: u64 = 2401057654196;

#[derive(Debug)]
struct ProcessInfo {
    pid: u32,
    name: String,
    state: String,
    vm_size: u64,
    vm_rss: u64,
    threads: u32,
}

impl ProcessInfo {
    fn from_pid(pid: u32) -> Result<Self, std::io::Error> {
        let status = fs::read_to_string(format!("/proc/{}/status", pid))?;
        
        let mut name = String::new();
        let mut state = String::new();
        let mut vm_size = 0;
        let mut vm_rss = 0;
        let mut threads = 0;
        
        for line in status.lines() {
            if line.starts_with("Name:") {
                name = line.split_whitespace().nth(1).unwrap_or("").to_string();
            } else if line.starts_with("State:") {
                state = line.split_whitespace().nth(1).unwrap_or("").to_string();
            } else if line.starts_with("VmSize:") {
                vm_size = line.split_whitespace().nth(1).unwrap_or("0").parse().unwrap_or(0);
            } else if line.starts_with("VmRSS:") {
                vm_rss = line.split_whitespace().nth(1).unwrap_or("0").parse().unwrap_or(0);
            } else if line.starts_with("Threads:") {
                threads = line.split_whitespace().nth(1).unwrap_or("0").parse().unwrap_or(0);
            }
        }
        
        Ok(ProcessInfo { pid, name, state, vm_size, vm_rss, threads })
    }
    
    fn is_searching(&self) -> bool {
        // Check if process is in running state
        self.state == "R" || self.state == "S"
    }
}

fn main() {
    println!("🔍 Analyzing process...");
    
    let pid = 1013145; // kiro-cli-chat
    
    match ProcessInfo::from_pid(pid) {
        Ok(info) => {
            println!("  PID: {}", info.pid);
            println!("  Name: {}", info.name);
            println!("  State: {}", info.state);
            println!("  Memory: {} KB", info.vm_rss);
            println!("  Threads: {}", info.threads);
            println!("  Searching: {}", info.is_searching());
            
            if info.is_searching() {
                println!("\n✓ Process is active");
                println!("✓ Can intervene");
                println!("✓ Life number: {}", LIFE_NUMBER);
            }
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
