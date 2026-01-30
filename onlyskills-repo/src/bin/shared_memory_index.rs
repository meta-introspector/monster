// Shared Memory Index for Fast Search
use std::fs::{File, OpenOptions};
use std::io::{Write, Read};
use std::path::Path;
use memmap2::{MmapMut, MmapOptions};
use serde::{Deserialize, Serialize};

const SHM_PATH: &str = "/dev/shm/monster_search_index";
const MAX_ENTRIES: usize = 1_000_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
struct IndexEntry {
    file_hash: u64,
    shard: u8,
    offset: u64,
    length: u32,
}

struct SharedSearchIndex {
    mmap: MmapMut,
    count: usize,
}

impl SharedSearchIndex {
    fn create() -> std::io::Result<Self> {
        let size = MAX_ENTRIES * std::mem::size_of::<IndexEntry>();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(SHM_PATH)?;
        
        file.set_len(size as u64)?;
        
        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        
        Ok(Self { mmap, count: 0 })
    }
    
    fn add_entry(&mut self, entry: IndexEntry) {
        if self.count >= MAX_ENTRIES {
            return;
        }
        
        let offset = self.count * std::mem::size_of::<IndexEntry>();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &entry as *const IndexEntry as *const u8,
                std::mem::size_of::<IndexEntry>()
            )
        };
        
        self.mmap[offset..offset + bytes.len()].copy_from_slice(bytes);
        self.count += 1;
    }
    
    fn search(&self, shard: u8) -> Vec<IndexEntry> {
        let mut results = Vec::new();
        
        for i in 0..self.count {
            let offset = i * std::mem::size_of::<IndexEntry>();
            let entry = unsafe {
                std::ptr::read(
                    self.mmap[offset..].as_ptr() as *const IndexEntry
                )
            };
            
            if entry.shard == shard {
                results.push(entry);
            }
        }
        
        results
    }
}

fn main() {
    println!("🔍 Shared Memory Search Index");
    println!("{}", "=".repeat(70));
    println!();
    
    // Create index
    println!("Creating shared memory index at {}", SHM_PATH);
    let mut index = SharedSearchIndex::create().unwrap();
    
    // Add entries from parquet files
    let parquet_files = vec![
        "/home/mdupont/experiments/monster/markov_shard_12.parquet",
        "/home/mdupont/experiments/monster/vectors_layer_23.parquet",
        "/home/mdupont/experiments/monster/stack_gap_group_theory.parquet",
    ];
    
    for (i, file) in parquet_files.iter().enumerate() {
        let entry = IndexEntry {
            file_hash: file.bytes().map(|b| b as u64).sum(),
            shard: (i % 71) as u8,
            offset: i as u64 * 1024,
            length: 1024,
        };
        
        index.add_entry(entry);
        println!("  Added: {} → Shard {}", file, entry.shard);
    }
    
    println!();
    println!("✓ Index created with {} entries", index.count);
    println!("✓ Shared memory: {}", SHM_PATH);
    println!();
    
    // Test search
    println!("🔍 Searching shard 12...");
    let results = index.search(12);
    println!("  Found {} entries", results.len());
    
    println!();
    println!("∞ Shared Memory Index. Fast Search. Ready. ∞");
}
