// Rust version of create_monster_autoencoder.py
// 71-layer autoencoder with Monster prime architecture

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

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
}

impl<B: Backend> MonsterAutoencoder<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            // Encoder
            encoder1: LinearConfig::new(5, 11).init(device),
            encoder2: LinearConfig::new(11, 23).init(device),
            encoder3: LinearConfig::new(23, 47).init(device),
            encoder4: LinearConfig::new(47, 71).init(device),
            
            // Decoder
            decoder1: LinearConfig::new(71, 47).init(device),
            decoder2: LinearConfig::new(47, 23).init(device),
            decoder3: LinearConfig::new(23, 11).init(device),
            decoder4: LinearConfig::new(11, 5).init(device),
        }
    }
    
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        // Encode
        let x = self.encoder1.forward(input);
        let x = x.relu();
        let x = self.encoder2.forward(x);
        let x = x.relu();
        let x = self.encoder3.forward(x);
        let x = x.relu();
        let latent = self.encoder4.forward(x);
        let latent = latent.relu();
        
        // Decode
        let x = self.decoder1.forward(latent);
        let x = x.relu();
        let x = self.decoder2.forward(x);
        let x = x.relu();
        let x = self.decoder3.forward(x);
        let x = x.relu();
        let output = self.decoder4.forward(x);
        
        output
    }
    
    pub fn encode(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.encoder1.forward(input);
        let x = x.relu();
        let x = self.encoder2.forward(x);
        let x = x.relu();
        let x = self.encoder3.forward(x);
        let x = x.relu();
        let latent = self.encoder4.forward(x);
        latent.relu()
    }
}

fn main() {
    println!("🧠 MONSTER AUTOENCODER");
    println!("{}", "=".repeat(70));
    println!("Architecture: 5 → 11 → 23 → 47 → 71 → 47 → 23 → 11 → 5");
    println!("{}", "=".repeat(70));
    println!();
    
    type MyBackend = burn::backend::Wgpu;
    let device = Default::default();
    
    let model = MonsterAutoencoder::<MyBackend>::new(&device);
    
    // Test forward pass
    let batch_size = 32;
    let input = Tensor::<MyBackend, 2>::random([batch_size, 5], burn::tensor::Distribution::Uniform(0.0, 1.0), &device);
    
    println!("Input shape: {:?}", input.dims());
    
    let output = model.forward(input.clone());
    println!("Output shape: {:?}", output.dims());
    
    let latent = model.encode(input);
    println!("Latent shape: {:?}", latent.dims());
    
    println!();
    println!("✅ Monster autoencoder initialized");
    println!("📊 Total layers: 8 (4 encoder + 4 decoder)");
    println!("🔢 Latent dimension: 71 (largest Monster prime)");
}
