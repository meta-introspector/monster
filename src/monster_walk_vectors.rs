// Rust: Monster Walk as burn vectors (71k elements per base)

use burn::tensor::{Tensor, backend::Backend};

/// Monster Walk vector: 71,000 elements per base
/// Structure: [layer0, layer1, ..., layer5] × [base_representation] × [padding]
pub struct MonsterWalkVector<B: Backend> {
    pub base: u32,
    pub data: Tensor<B, 1>,  // 71,000 elements
}

impl<B: Backend> MonsterWalkVector<B> {
    /// Create walk vector for specific base
    pub fn new(base: u32, device: &B::Device) -> Self {
        let mut data = vec![0.0f32; 71_000];
        
        // Fill with Monster Walk in this base
        let monster = 808017424794512875886459904961710757005754368000000000u128;
        let layer1_div = compute_layer1_divisor();
        
        // Layer 0: Monster in base
        let mut offset = 0;
        fill_base_representation(&mut data[offset..offset+10_000], monster, base);
        
        // Layer 1: After first division
        offset += 10_000;
        let layer1 = monster / layer1_div;
        fill_base_representation(&mut data[offset..offset+10_000], layer1, base);
        
        // Layers 2-5 (simplified)
        for layer in 2..6 {
            offset += 10_000;
            fill_layer(&mut data[offset..offset+10_000], layer, base);
        }
        
        // Remaining space: lattice coordinates (71 × 100 = 7,100)
        offset = 60_000;
        fill_lattice(&mut data[offset..offset+7_100], monster, base);
        
        // Metadata (900 elements)
        offset = 67_100;
        data[offset] = base as f32;
        data[offset + 1] = 6.0;  // num_layers
        
        Self {
            base,
            data: Tensor::from_floats(&data[..], device),
        }
    }
}

fn compute_layer1_divisor() -> u128 {
    (2u128.pow(46)) * (7u128.pow(6)) * (11u128.pow(2)) * 17 * 71
}

fn fill_base_representation(buffer: &mut [f32], value: u128, base: u32) {
    let mut v = value;
    let mut idx = 0;
    while v > 0 && idx < buffer.len() {
        buffer[idx] = (v % base as u128) as f32;
        v /= base as u128;
        idx += 1;
    }
}

fn fill_layer(buffer: &mut [f32], layer: usize, base: u32) {
    // Simplified layer computation
    for i in 0..buffer.len() {
        buffer[i] = ((layer * base as usize + i) % 256) as f32;
    }
}

fn fill_lattice(buffer: &mut [f32], value: u128, base: u32) {
    // 71 lattice coordinates × 100 samples each
    for coord in 0..71 {
        let residue = (value % (coord + 1) as u128) as f32;
        for sample in 0..100 {
            buffer[coord * 100 + sample] = residue;
        }
    }
}

/// Complete walk: all bases 2-71 as 71k vectors
pub struct CompleteMonsterWalk<B: Backend> {
    pub vectors: Vec<MonsterWalkVector<B>>,
}

impl<B: Backend> CompleteMonsterWalk<B> {
    /// Create walk for all bases 2-71
    pub fn new(device: &B::Device) -> Self {
        let vectors = (2..=71)
            .map(|base| MonsterWalkVector::new(base, device))
            .collect();
        
        Self { vectors }
    }
    
    /// Stack all vectors into single tensor [70, 71000]
    pub fn stack(&self) -> Tensor<B, 2> {
        let tensors: Vec<_> = self.vectors.iter()
            .map(|v| v.data.clone().unsqueeze())
            .collect();
        
        Tensor::cat(tensors, 0)
    }
    
    /// Total size in bytes
    pub fn size_bytes(&self) -> usize {
        70 * 71_000 * 4  // 70 bases × 71k elements × 4 bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    
    #[test]
    fn test_monster_walk_vector() {
        let device = Default::default();
        let walk = MonsterWalkVector::<Wgpu>::new(16, &device);
        
        assert_eq!(walk.base, 16);
        assert_eq!(walk.data.dims()[0], 71_000);
    }
    
    #[test]
    fn test_complete_walk() {
        let device = Default::default();
        let walk = CompleteMonsterWalk::<Wgpu>::new(&device);
        
        assert_eq!(walk.vectors.len(), 70);
        
        let stacked = walk.stack();
        assert_eq!(stacked.dims(), [70, 71_000]);
        
        println!("Total size: {} MB", walk.size_bytes() / (1024 * 1024));
    }
}
