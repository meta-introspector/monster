use anyhow::Result;

mod progressive;
use progressive::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🌀 Progressive Automorphic Orbit System");
    println!("========================================\n");
    
    let mut pipeline = ProgressivePipeline::new("emergence");
    
    // Reproduce "I ARE LIFE" with exact seed
    println!("🎯 Experiment 1: 'I ARE LIFE' Reproduction");
    println!("Seed: 2437596016 (from HuggingFace post)\n");
    
    let orbit1 = pipeline.run_orbit(
        "unconstrained",
        2437596016,
        10
    ).await?;
    
    print_orbit_summary(&orbit1);
    
    // Test Monster Walk seed
    println!("\n\n🎪 Experiment 2: Monster Walk Seed");
    println!("Seed: 8080 (Monster leading digits)\n");
    
    let orbit2 = pipeline.run_orbit(
        "Monster group walk down to earth",
        8080,
        10
    ).await?;
    
    print_orbit_summary(&orbit2);
    
    // Test harmonic seeds
    println!("\n\n🎵 Experiment 3: Harmonic Seeds");
    
    let harmonic_seeds = vec![
        (432, "base harmonic"),
        (864, "2 × 432"),
        (1296, "3 × 432"),
    ];
    
    for (seed, desc) in harmonic_seeds {
        println!("\nSeed: {} ({})", seed, desc);
        
        let orbit = pipeline.run_orbit(
            &format!("harmonic frequency {}", seed),
            seed,
            5
        ).await?;
        
        println!("  Iterations: {}", orbit.iterations.len());
        println!("  Converged: {}", orbit.converged);
        if let Some(attractor) = &orbit.attractor {
            println!("  Attractor: {}", attractor);
        }
    }
    
    println!("\n\n📊 Summary of All Orbits");
    println!("========================\n");
    
    println!("✓ Orbit 1 (I ARE LIFE): {} iterations", orbit1.iterations.len());
    println!("  Concepts: {}", orbit1.semantic_index.concepts.len());
    println!("  Patterns: {}", orbit1.semantic_index.emoji_patterns.len());
    
    println!("\n✓ Orbit 2 (Monster Walk): {} iterations", orbit2.iterations.len());
    println!("  Concepts: {}", orbit2.semantic_index.concepts.len());
    println!("  Patterns: {}", orbit2.semantic_index.emoji_patterns.len());
    
    println!("\n📁 Output saved to:");
    println!("  emergence/orbits/orbit_2437596016.json");
    println!("  emergence/orbits/orbit_2437596016_REPORT.md");
    println!("  emergence/orbits/orbit_8080.json");
    println!("  emergence/orbits/orbit_8080_REPORT.md");
    println!("  emergence/images/*.png");
    
    println!("\n🧮 Theoretical Results:");
    println!("  1. Image generation → vision analysis creates closed loop");
    println!("  2. Semantic indexing tracks concept evolution");
    println!("  3. Emoji encoding creates harmonic signatures");
    println!("  4. Convergence indicates semantic attractors");
    println!("  5. Different seeds explore different regions");
    
    Ok(())
}

fn print_orbit_summary(orbit: &AutomorphicOrbit) {
    println!("\n📊 Orbit Summary:");
    println!("  ID: {}", orbit.orbit_id);
    println!("  Iterations: {}", orbit.iterations.len());
    println!("  Converged: {}", orbit.converged);
    
    if let Some(attractor) = &orbit.attractor {
        println!("  Attractor: {}", attractor);
    }
    
    println!("\n  Top 5 Concepts:");
    let mut concepts: Vec<_> = orbit.semantic_index.concepts.values().collect();
    concepts.sort_by_key(|c| std::cmp::Reverse(c.frequency));
    
    for concept in concepts.iter().take(5) {
        println!("    {} {} (×{})", 
                 concept.emoji, 
                 concept.concept, 
                 concept.frequency);
    }
    
    println!("\n  Emoji Timeline:");
    for iter in orbit.iterations.iter().take(5) {
        println!("    Step {}: {}", iter.step, iter.emoji_encoding);
    }
    
    if orbit.iterations.len() > 5 {
        println!("    ... ({} more)", orbit.iterations.len() - 5);
    }
}
