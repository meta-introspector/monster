// Rust version of train_monster.py
// Train Monster autoencoder on LMFDB shards

use burn::{
    module::Module,
    nn::loss::MseLoss,
    optim::{AdamConfig, Optimizer},
    tensor::{backend::AutodiffBackend, Tensor},
    train::{TrainOutput, TrainStep, ValidStep},
};
use polars::prelude::*;
use std::fs::File;

// Import from create_monster_autoencoder
use crate::MonsterAutoencoder;

struct MonsterBatch<B: AutodiffBackend> {
    input: Tensor<B, 2>,
    target: Tensor<B, 2>,
}

impl<B: AutodiffBackend> TrainStep<MonsterBatch<B>, MseLoss<B>> for MonsterAutoencoder<B> {
    fn step(&self, batch: MonsterBatch<B>) -> TrainOutput<MseLoss<B>> {
        let output = self.forward(batch.input.clone());
        let loss = MseLoss::new().forward(output, batch.target);
        
        TrainOutput::new(self, loss.backward(), loss)
    }
}

impl<B: AutodiffBackend> ValidStep<MonsterBatch<B>, MseLoss<B>> for MonsterAutoencoder<B> {
    fn step(&self, batch: MonsterBatch<B>) -> MseLoss<B> {
        let output = self.forward(batch.input.clone());
        MseLoss::new().forward(output, batch.target)
    }
}

fn load_lmfdb_shard(shard_id: u8) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let path = format!("lmfdb_shard_{:02}.parquet", shard_id);
    let file = File::open(path)?;
    let df = ParquetReader::new(file).finish()?;
    
    // Extract features (first 5 columns)
    let mut data = Vec::new();
    
    for row in 0..df.height().min(1000) {
        let mut features = Vec::new();
        for col_idx in 0..5.min(df.width()) {
            let col = df.get_columns()[col_idx];
            let val = match col.get(row) {
                Ok(v) => format!("{:?}", v).len() as f32,
                Err(_) => 0.0,
            };
            features.push(val);
        }
        if features.len() == 5 {
            data.push(features);
        }
    }
    
    Ok(data)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏋️ TRAINING MONSTER AUTOENCODER");
    println!("{}", "=".repeat(70));
    println!();
    
    type MyBackend = burn::backend::Autodiff<burn::backend::Wgpu>;
    let device = Default::default();
    
    let model = MonsterAutoencoder::<MyBackend>::new(&device);
    let mut optim = AdamConfig::new().init();
    
    let epochs = 10;
    let batch_size = 32;
    
    for epoch in 0..epochs {
        println!("Epoch {}/{}", epoch + 1, epochs);
        
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        
        // Train on each shard
        for shard_id in 0..71 {
            let data = match load_lmfdb_shard(shard_id) {
                Ok(d) => d,
                Err(_) => continue,
            };
            
            if data.is_empty() {
                continue;
            }
            
            // Create batches
            for batch_start in (0..data.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(data.len());
                let batch_data: Vec<Vec<f32>> = data[batch_start..batch_end].to_vec();
                
                if batch_data.len() < batch_size {
                    continue;
                }
                
                // Convert to tensors
                let flat: Vec<f32> = batch_data.iter().flatten().copied().collect();
                let input = Tensor::<MyBackend, 2>::from_floats(
                    flat.as_slice(),
                    &device
                ).reshape([batch_size, 5]);
                
                let batch = MonsterBatch {
                    input: input.clone(),
                    target: input.clone(),
                };
                
                let output = model.step(batch);
                total_loss += output.loss.into_scalar();
                num_batches += 1;
                
                // Update weights
                model = optim.step(1e-3, model, output.grads);
            }
        }
        
        let avg_loss = total_loss / num_batches as f32;
        println!("  Average loss: {:.6}", avg_loss);
    }
    
    println!();
    println!("✅ Training complete");
    
    Ok(())
}
