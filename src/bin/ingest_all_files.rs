// Updated: Collect ALL files to parquet with Monster sharding

use std::process::Command;
use std::io::{BufRead, BufReader};
use std::fs;
use std::sync::Arc;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use arrow::array::{StringArray, UInt64Array, UInt8Array};
use arrow::record_batch::RecordBatch;
use arrow::datatypes::{Schema, Field, DataType};

const MONSTER_PRIMES: [u32; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 INGESTING ALL FILES WITH MONSTER SHARDING");
    println!("{}", "=".repeat(70));
    println!();
    
    // Run plocate for all files
    let output = Command::new("plocate")
        .args([""])
        .output()?;
    
    let reader = BufReader::new(&output.stdout[..]);
    let mut paths = Vec::new();
    let mut sizes = Vec::new();
    let mut shards = Vec::new();
    let mut primes = Vec::new();
    
    let mut count = 0;
    for line in reader.lines().filter_map(|l| l.ok()) {
        let size = fs::metadata(&line).map(|m| m.len()).unwrap_or(0);
        
        // Hash to Monster shard
        let hash = line.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let shard = (hash % 15) as u8;
        let prime = MONSTER_PRIMES[shard as usize];
        
        paths.push(line);
        sizes.push(size);
        shards.push(shard);
        primes.push(prime);
        
        count += 1;
        if count % 100000 == 0 {
            println!("  Processed: {} files", count);
        }
    }
    
    println!();
    println!("✅ Total files: {}", count);
    println!();
    
    // Create parquet
    let schema = Arc::new(Schema::new(vec![
        Field::new("file_path", DataType::Utf8, false),
        Field::new("file_size", DataType::UInt64, false),
        Field::new("shard", DataType::UInt8, false),
        Field::new("prime_label", DataType::UInt32, false),
    ]));
    
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(paths)),
            Arc::new(UInt64Array::from(sizes)),
            Arc::new(UInt8Array::from(shards)),
            Arc::new(arrow::array::UInt32Array::from(primes)),
        ],
    )?;
    
    let file = fs::File::create("all_files_monster_shards.parquet")?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    
    println!("✅ Created: all_files_monster_shards.parquet");
    println!("   Rows: {}", count);
    println!();
    println!("{}", "=".repeat(70));
    println!("✅ All files ingested with Monster sharding!");
    
    Ok(())
}
