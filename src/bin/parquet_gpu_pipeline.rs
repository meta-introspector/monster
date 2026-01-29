// Rust: Parquet pipeline to GPU in shards (optimal speed)

use polars::prelude::*;
use tokio::sync::mpsc;
use std::path::PathBuf;
use burn::tensor::{Tensor, backend::Backend, Data};

/// Parquet shard reader
pub struct ParquetShardReader {
    shard_paths: Vec<PathBuf>,
    chunk_size: usize,
}

impl ParquetShardReader {
    pub fn new(shard_dir: PathBuf, chunk_size: usize) -> Self {
        let shard_paths = std::fs::read_dir(shard_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "parquet"))
            .map(|e| e.path())
            .collect();
        
        Self { shard_paths, chunk_size }
    }
    
    /// Stream shards to GPU pipeline
    pub async fn stream_to_gpu<B: Backend>(
        &self,
        device: &B::Device,
    ) -> mpsc::Receiver<Tensor<B, 2>> {
        let (tx, rx) = mpsc::channel(16);  // Buffer 16 batches
        let paths = self.shard_paths.clone();
        let chunk_size = self.chunk_size;
        let device = device.clone();
        
        tokio::spawn(async move {
            for path in paths {
                // Read parquet in chunks
                let df = ParquetReader::new(std::fs::File::open(&path).unwrap())
                    .finish()
                    .unwrap();
                
                // Convert to tensors in chunks
                for chunk in df.iter_chunks(chunk_size) {
                    let tensor = Self::df_to_tensor::<B>(&chunk, &device);
                    if tx.send(tensor).await.is_err() {
                        break;
                    }
                }
            }
        });
        
        rx
    }
    
    fn df_to_tensor<B: Backend>(df: &DataFrame, device: &B::Device) -> Tensor<B, 2> {
        let rows = df.height();
        let cols = df.width();
        
        // Convert to flat f32 array
        let mut data = Vec::with_capacity(rows * cols);
        for col in df.get_columns() {
            if let Ok(series) = col.cast(&DataType::Float32) {
                let values = series.f32().unwrap();
                data.extend(values.into_iter().map(|v| v.unwrap_or(0.0)));
            }
        }
        
        Tensor::from_data(Data::new(data, [rows, cols].into()), device)
    }
}

/// GPU pipeline processor
pub struct GPUPipeline<B: Backend> {
    device: B::Device,
    batch_size: usize,
}

impl<B: Backend> GPUPipeline<B> {
    pub fn new(device: B::Device, batch_size: usize) -> Self {
        Self { device, batch_size }
    }
    
    /// Process stream of tensors
    pub async fn process_stream(
        &self,
        mut rx: mpsc::Receiver<Tensor<B, 2>>,
    ) -> Vec<Tensor<B, 2>> {
        let mut results = Vec::new();
        let mut batch = Vec::new();
        
        while let Some(tensor) = rx.recv().await {
            batch.push(tensor);
            
            if batch.len() >= self.batch_size {
                let processed = self.process_batch(&batch);
                results.push(processed);
                batch.clear();
            }
        }
        
        // Process remaining
        if !batch.is_empty() {
            results.push(self.process_batch(&batch));
        }
        
        results
    }
    
    fn process_batch(&self, batch: &[Tensor<B, 2>]) -> Tensor<B, 2> {
        // Stack and process on GPU
        let stacked = Tensor::cat(batch.to_vec(), 0);
        
        // Apply Monster Walk transformation
        stacked * 71.0  // Multiply by 71 (Monster prime)
    }
}

/// Complete pipeline: Parquet → GPU → Results
pub async fn run_parquet_pipeline<B: Backend>(
    shard_dir: PathBuf,
    device: B::Device,
) -> Result<Vec<Tensor<B, 2>>, Box<dyn std::error::Error>> {
    println!("🚀 Parquet → GPU Pipeline");
    println!("="*70);
    println!();
    
    // Initialize reader
    let reader = ParquetShardReader::new(shard_dir.clone(), 10_000);
    println!("✓ Found {} shards", reader.shard_paths.len());
    
    // Start streaming
    let rx = reader.stream_to_gpu::<B>(&device).await;
    println!("✓ Streaming started");
    
    // Process on GPU
    let pipeline = GPUPipeline::new(device, 32);
    let results = pipeline.process_stream(rx).await;
    
    println!("✓ Processed {} batches", results.len());
    println!();
    println!("="*70);
    println!("✅ Pipeline complete!");
    
    Ok(results)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use burn::backend::Wgpu;
    
    let shard_dir = PathBuf::from("shards");
    let device = Default::default();
    
    let results = run_parquet_pipeline::<Wgpu>(shard_dir, device).await?;
    
    println!("\nResults: {} tensors", results.len());
    
    Ok(())
}
