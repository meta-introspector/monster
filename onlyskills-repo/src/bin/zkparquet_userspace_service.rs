// ZK Parquet Userspace Service - Query shared memory
use memmap2::Mmap;
use std::fs::File;

const SHM_PATH: &str = "/dev/shm/monster_zkparquet";

#[derive(Debug)]
struct ZKParquetRow {
    inode: u64,
    zk71_shard: u8,
    security_orbit: u8,
    timestamp: i64,
}

fn read_zkparquet_table() -> std::io::Result<Vec<ZKParquetRow>> {
    let file = File::open(SHM_PATH)?;
    let mmap = unsafe { Mmap::map(&file)? };
    
    // Read header
    let count = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    
    let mut rows = Vec::with_capacity(count);
    let mut offset = 8;
    
    for _ in 0..count {
        if offset + 32 > mmap.len() { break; }
        
        let inode = u64::from_le_bytes(mmap[offset..offset+8].try_into().unwrap());
        let zk71_shard = mmap[offset+8];
        let security_orbit = mmap[offset+9];
        let timestamp = i64::from_le_bytes(mmap[offset+10..offset+18].try_into().unwrap());
        
        rows.push(ZKParquetRow {
            inode,
            zk71_shard,
            security_orbit,
            timestamp,
        });
        
        offset += 32;
    }
    
    Ok(rows)
}

fn query_by_shard(shard: u8) -> std::io::Result<Vec<ZKParquetRow>> {
    let rows = read_zkparquet_table()?;
    Ok(rows.into_iter().filter(|r| r.zk71_shard == shard).collect())
}

fn query_by_security_orbit(orbit: u8) -> std::io::Result<Vec<ZKParquetRow>> {
    let rows = read_zkparquet_table()?;
    Ok(rows.into_iter().filter(|r| r.security_orbit == orbit).collect())
}

fn main() {
    println!("🔍 ZK Parquet Userspace Service");
    println!("{}", "=".repeat(70));
    println!();
    
    match read_zkparquet_table() {
        Ok(rows) => {
            println!("✓ Read {} rows from shared memory", rows.len());
            println!();
            
            // Show sample
            println!("📋 Sample rows:");
            for row in rows.iter().take(5) {
                println!("  Inode: {}, ZK71 Shard: {}, Security Orbit: {}, Timestamp: {}",
                    row.inode, row.zk71_shard, row.security_orbit, row.timestamp);
            }
            println!();
            
            // Statistics by shard
            println!("📊 Distribution by ZK71 Shard:");
            for shard in [11, 23, 31, 47, 59, 71] {
                let count = rows.iter().filter(|r| r.zk71_shard == shard).count();
                println!("  Shard {}: {} rows", shard, count);
            }
            println!();
            
            // Statistics by security orbit
            println!("🔐 Distribution by Security Orbit:");
            for orbit in [11, 23, 31, 47, 59, 71] {
                let count = rows.iter().filter(|r| r.security_orbit == orbit).count();
                println!("  Orbit {}: {} rows", orbit, count);
            }
            println!();
            
            // Query examples
            println!("🔎 Query Examples:");
            
            if let Ok(shard_71) = query_by_shard(71) {
                println!("  Shard 71: {} rows", shard_71.len());
            }
            
            if let Ok(orbit_11) = query_by_security_orbit(11) {
                println!("  Orbit 11 (safe): {} rows", orbit_11.len());
            }
            
            if let Ok(orbit_71) = query_by_security_orbit(71) {
                println!("  Orbit 71 (vile): {} rows", orbit_71.len());
            }
        }
        Err(e) => {
            eprintln!("✗ Failed to read shared memory: {}", e);
            eprintln!("  Make sure zkparquet_kernel_server is running");
        }
    }
    
    println!();
    println!("∞ ZK Parquet. Shared Memory. Security Orbits. ∞");
}
