# Best Practices: Parallel Parquet Processing from .txt File List

Based on crossbeam patterns from meta-introspector codebase.

## Core Pattern: Producer-Consumer with Bounded Channels

```rust
use crossbeam::channel::{bounded, Sender, Receiver};
use std::thread;
use std::sync::{Arc, Mutex};
use parquet::arrow::ArrowWriter;
use arrow::record_batch::RecordBatch;

const NUM_WORKERS: usize = 20;
const CHANNEL_SIZE: usize = 1000;
const BATCH_SIZE: usize = 10000;
```

## Architecture

```
[File List Reader] → [Bounded Channel] → [20 Workers] → [Bounded Channel] → [Single Writer Thread]
```

## Implementation

### 1. Read File List and Queue Work

```rust
fn queue_parquet_files(file_list_path: &str, sender: Sender<String>) -> Result<usize> {
    let content = fs::read_to_string(file_list_path)?;
    let mut count = 0;
    
    for line in content.lines() {
        let path = line.trim();
        if !path.is_empty() && path.ends_with(".parquet") {
            sender.send(path.to_string())?;
            count += 1;
            
            if count % 1000 == 0 {
                println!("Queued {} files", count);
            }
        }
    }
    
    println!("Total files queued: {}", count);
    Ok(count)
}
```

### 2. Worker Pattern (Process Parquet Files)

```rust
fn spawn_workers(
    receiver: Receiver<String>,
    results_sender: Sender<Vec<YourDataType>>,
    num_workers: usize,
) {
    for worker_id in 0..num_workers {
        let rx = receiver.clone();
        let tx = results_sender.clone();
        
        thread::spawn(move || {
            while let Ok(parquet_path) = rx.recv() {
                match process_parquet_file(&parquet_path) {
                    Ok(data) => {
                        if !data.is_empty() {
                            let _ = tx.send(data);
                        }
                    }
                    Err(e) => {
                        eprintln!("Worker {}: Error processing {}: {}", 
                            worker_id, parquet_path, e);
                    }
                }
            }
        });
    }
}

fn process_parquet_file(path: &str) -> Result<Vec<YourDataType>> {
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(8192)
        .build()?;
    
    let mut results = Vec::new();
    
    for batch_result in reader {
        let batch = batch_result?;
        // Extract and transform data
        results.extend(extract_from_batch(&batch)?);
    }
    
    Ok(results)
}
```

### 3. Writer Pattern (Single Thread)

```rust
fn spawn_writer_thread(
    receiver: Receiver<Vec<YourDataType>>,
    output_path: &str,
    schema: Arc<Schema>,
) -> thread::JoinHandle<Result<usize>> {
    let output_path = output_path.to_string();
    
    thread::spawn(move || {
        let file = File::create(&output_path)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
        let mut total_rows = 0;
        let mut buffer = Vec::new();
        
        while let Ok(data) = receiver.recv() {
            buffer.extend(data);
            
            if buffer.len() >= BATCH_SIZE {
                let batch = create_record_batch(&schema, &buffer)?;
                writer.write(&batch)?;
                total_rows += buffer.len();
                
                println!("Wrote batch - total: {} rows", total_rows);
                buffer.clear();
            }
        }
        
        // Write remaining
        if !buffer.is_empty() {
            let batch = create_record_batch(&schema, &buffer)?;
            writer.write(&batch)?;
            total_rows += buffer.len();
        }
        
        writer.close()?;
        Ok(total_rows)
    })
}
```

### 4. Main Orchestration

```rust
fn main() -> Result<()> {
    let file_list = "parquet_files.txt";
    let output = "output.parquet";
    
    // Create channels
    let (work_tx, work_rx) = bounded(CHANNEL_SIZE);
    let (result_tx, result_rx) = bounded(100);
    
    // Spawn workers
    spawn_workers(work_rx, result_tx.clone(), NUM_WORKERS);
    
    // Spawn writer
    let schema = create_schema();
    let writer_handle = spawn_writer_thread(result_rx, output, schema);
    
    // Queue work
    let total_files = queue_parquet_files(file_list, work_tx.clone())?;
    drop(work_tx); // Signal workers to finish
    drop(result_tx); // Signal writer to finish
    
    // Wait for completion
    let total_rows = writer_handle.join().unwrap()?;
    
    println!("✅ Processed {} files, wrote {} rows", total_files, total_rows);
    Ok(())
}
```

## Key Principles

### 1. **Bounded Channels**
- Use `bounded(1000)` for work queue (prevents memory overflow)
- Use `bounded(100)` for results (backpressure on workers)

### 2. **Single Writer Thread**
- Only ONE thread writes to output parquet
- Prevents file corruption and lock contention
- Batches writes for efficiency

### 3. **Worker Count**
- 20 workers for I/O-bound tasks (reading parquet files)
- Adjust based on: CPU cores, disk I/O, network bandwidth

### 4. **Batch Size**
- 10,000 rows per parquet write batch
- Balance memory usage vs write efficiency

### 5. **Error Handling**
- Workers log errors but continue processing
- Track skipped files for later review
- Use `Arc<Mutex<Vec<String>>>` for shared error log

### 6. **Progress Reporting**
```rust
if count % 1000 == 0 {
    println!("Progress: {} files processed", count);
}
```

### 7. **Graceful Shutdown**
```rust
drop(work_tx);    // Close work channel → workers exit
drop(result_tx);  // Close result channel → writer exits
writer_handle.join().unwrap()?; // Wait for writer
```

## Memory Management

### For Large Files
```rust
// Stream processing - don't load entire file
let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
    .with_batch_size(8192)  // Process in chunks
    .build()?;

for batch in reader {
    process_batch(batch?)?;
}
```

### For Many Small Files
```rust
// Buffer results before sending
let mut buffer = Vec::with_capacity(1000);
for item in items {
    buffer.push(item);
    if buffer.len() >= 1000 {
        result_tx.send(buffer.clone())?;
        buffer.clear();
    }
}
```

## Performance Tuning

### Disk I/O Bound
- Increase workers (20-40)
- Reduce batch size
- Use SSD storage

### CPU Bound
- Match worker count to CPU cores
- Increase batch size
- Use SIMD operations

### Memory Bound
- Reduce worker count
- Reduce batch size
- Stream processing only

## Example: Monster Project Use Case

```rust
// Process 71³ = 357,911 parquet files
fn process_monster_shards() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("shard_id", DataType::UInt32, false),
        Field::new("witness_count", DataType::UInt32, false),
        Field::new("frequency", DataType::Float64, false),
        Field::new("lmfdb_signature", DataType::Utf8, false),
    ]));
    
    let (work_tx, work_rx) = bounded(1000);
    let (result_tx, result_rx) = bounded(100);
    
    spawn_workers(work_rx, result_tx.clone(), 20);
    let writer = spawn_writer_thread(result_rx, "monster_71_cubed.parquet", schema);
    
    queue_parquet_files("lmfdb_shards.txt", work_tx)?;
    
    writer.join().unwrap()
}
```

## Monitoring

```rust
use std::time::Instant;

let start = Instant::now();
let total_rows = process_files()?;
let elapsed = start.elapsed();

println!("Throughput: {:.0} rows/sec", 
    total_rows as f64 / elapsed.as_secs_f64());
```

## Common Pitfalls

❌ **Multiple writers to same file** → Use single writer thread  
❌ **Unbounded channels** → Use `bounded()` for backpressure  
❌ **Loading entire file** → Stream with batch reader  
❌ **No error tracking** → Log skipped files  
❌ **Blocking on full channel** → Use `try_send()` with retry  

## References

- `crossbeam_repo_compressor.rs` - 20-worker compression pattern
- `nix2parquet.rs` - Parquet streaming with crossbeam
- `crossbeam_rustc_analyzer_complete.rs` - File analysis pattern
- `https_commit_fetcher.rs` - Producer-consumer with results collection
