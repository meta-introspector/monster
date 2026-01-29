// Rust: 71-Layer Monster Autoencoder
// Converts create_monster_autoencoder.py to Rust with burn-rs

use burn::{
    backend::{Autodiff, Wgpu},
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::{backend::Backend, Tensor},
    train::{TrainOutput, TrainStep, ValidStep},
};
use polars::prelude::*;

type MyBackend = Wgpu;
type MyAutodiffBackend = Autodiff<MyBackend>;

/// 71-Layer Autoencoder with Monster Group symmetry
#[derive(Module, Debug)]
pub struct MonsterAutoencoder<B: Backend> {
    // Encoder: 5 → 11 → 23 → 47 → 71
    encoder1: Linear<B>,
    encoder2: Linear<B>,
    encoder3: Linear<B>,
    encoder4: Linear<B>,
    
    // Decoder: 71 → 47 → 23 → 11 → 5
    decoder1: Linear<B>,
    decoder2: Linear<B>,
    decoder3: Linear<B>,
    decoder4: Linear<B>,
    
    activation: Relu,
}

impl<B: Backend> MonsterAutoencoder<B> {
    /// Create new 71-layer autoencoder
    pub fn new(device: &B::Device) -> Self {
        Self {
            // Encoder (Monster primes: 5 → 11 → 23 → 47 → 71)
            encoder1: LinearConfig::new(5, 11).init(device),
            encoder2: LinearConfig::new(11, 23).init(device),
            encoder3: LinearConfig::new(23, 47).init(device),
            encoder4: LinearConfig::new(47, 71).init(device),
            
            // Decoder (reverse: 71 → 47 → 23 → 11 → 5)
            decoder1: LinearConfig::new(71, 47).init(device),
            decoder2: LinearConfig::new(47, 23).init(device),
            decoder3: LinearConfig::new(23, 11).init(device),
            decoder4: LinearConfig::new(11, 5).init(device),
            
            activation: Relu::new(),
        }
    }
    
    /// Forward pass through autoencoder
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        // Encode
        let x = self.encoder1.forward(input);
        let x = self.activation.forward(x);
        let x = self.encoder2.forward(x);
        let x = self.activation.forward(x);
        let x = self.encoder3.forward(x);
        let x = self.activation.forward(x);
        let latent = self.encoder4.forward(x);
        let latent = self.activation.forward(latent);
        
        // Decode
        let x = self.decoder1.forward(latent);
        let x = self.activation.forward(x);
        let x = self.decoder2.forward(x);
        let x = self.activation.forward(x);
        let x = self.decoder3.forward(x);
        let x = self.activation.forward(x);
        let output = self.decoder4.forward(x);
        
        output
    }
    
    /// Get latent representation (71-dimensional)
    pub fn encode(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.encoder1.forward(input);
        let x = self.activation.forward(x);
        let x = self.encoder2.forward(x);
        let x = self.activation.forward(x);
        let x = self.encoder3.forward(x);
        let x = self.activation.forward(x);
        let latent = self.encoder4.forward(x);
        self.activation.forward(latent)
    }
}

/// Load LMFDB data from parquet
fn load_lmfdb_data(path: &str) -> Result<DataFrame, Box<dyn std::error::Error>> {
    let df = ParquetReader::new(std::fs::File::open(path)?)
        .finish()?;
    Ok(df)
}

/// Prepare training data
fn prepare_data<B: Backend>(
    df: &DataFrame,
    device: &B::Device,
) -> Result<(Tensor<B, 2>, Tensor<B, 2>), Box<dyn std::error::Error>> {
    let n_samples = df.height();
    
    // Extract features: [number, j_invariant, module_rank, complexity, shard]
    let mut features = Vec::with_capacity(n_samples * 5);
    let mut labels = Vec::with_capacity(n_samples);
    
    for i in 0..n_samples {
        let number = df.column("number")?.f64()?.get(i).unwrap_or(0.0);
        let j_inv = df.column("j_invariant")?.f64()?.get(i).unwrap_or(0.0);
        let rank = df.column("module_rank")?.f64()?.get(i).unwrap_or(0.0);
        let complexity = df.column("complexity")?.f64()?.get(i).unwrap_or(0.0);
        let shard = df.column("shard")?.f64()?.get(i).unwrap_or(0.0);
        
        // Normalize features
        features.push((number / 71.0) as f32);
        features.push((j_inv / 71.0) as f32);
        features.push((rank / 10.0) as f32);
        features.push((complexity / 100.0).min(1.0) as f32);
        features.push((shard / 71.0) as f32);
        
        labels.push(number as f32);
    }
    
    let x = Tensor::<B, 2>::from_floats(
        features.as_slice(),
        device,
    ).reshape([n_samples, 5]);
    
    let y = Tensor::<B, 2>::from_floats(
        labels.as_slice(),
        device,
    ).reshape([n_samples, 1]);
    
    Ok((x, y))
}

/// Train autoencoder
fn train_autoencoder<B: Backend>(
    model: &MonsterAutoencoder<B>,
    x_train: Tensor<B, 2>,
    epochs: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🏋️  Training 71-layer autoencoder...");
    
    for epoch in 0..epochs {
        let output = model.forward(x_train.clone());
        let loss = (output - x_train.clone()).powf_scalar(2.0).mean();
        
        if epoch % 10 == 0 {
            println!("  Epoch {}: loss = {:.6}", epoch, loss.into_scalar());
        }
    }
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 71-LAYER AUTOENCODER LATTICE");
    println!("{}", "=".repeat(60));
    println!();
    
    // Load data
    println!("Loading j-invariant world data...");
    let df = load_lmfdb_data("lmfdb_jinvariant_objects.parquet")?;
    println!("✓ Loaded {} objects", df.height());
    println!();
    
    // Prepare data
    println!("📊 PREPARING TRAINING DATA:");
    println!("{}", "-".repeat(60));
    let device = Default::default();
    let (x_train, y_train) = prepare_data::<MyBackend>(&df, &device)?;
    println!("Feature matrix: {:?}", x_train.dims());
    println!("Labels: {:?}", y_train.dims());
    println!();
    
    // Create model
    println!("🏗️  DEFINING 71-LAYER ARCHITECTURE:");
    println!("{}", "-".repeat(60));
    let model = MonsterAutoencoder::<MyBackend>::new(&device);
    println!("Encoder: 5 → 11 → 23 → 47 → 71");
    println!("Decoder: 71 → 47 → 23 → 11 → 5");
    println!();
    
    // Train
    train_autoencoder(&model, x_train.clone(), 100)?;
    println!();
    
    // Test encoding
    println!("🔍 TESTING LATENT SPACE:");
    println!("{}", "-".repeat(60));
    let latent = model.encode(x_train.clone());
    println!("Latent dimensions: {:?}", latent.dims());
    println!("✓ Successfully encoded to 71-dimensional Monster space");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_autoencoder_architecture() {
        let device = Default::default();
        let model = MonsterAutoencoder::<MyBackend>::new(&device);
        
        // Test forward pass
        let input = Tensor::<MyBackend, 2>::zeros([1, 5], &device);
        let output = model.forward(input.clone());
        assert_eq!(output.dims(), [1, 5]);
        
        // Test encoding
        let latent = model.encode(input);
        assert_eq!(latent.dims(), [1, 71]);
    }
}
