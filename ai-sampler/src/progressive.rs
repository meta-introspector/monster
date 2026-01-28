use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::collections::HashMap;

mod model_lattice;
mod flux_integration;

use model_lattice::*;
use flux_integration::*;

/// Progressive automorphic orbit system
/// Image Gen → Vision → Feedback → Image Gen → ...
#[derive(Debug, Serialize, Clone)]
pub struct AutomorphicOrbit {
    pub orbit_id: String,
    pub initial_seed: u64,
    pub iterations: Vec<OrbitIteration>,
    pub semantic_index: SemanticIndex,
    pub model_scores: HashMap<String, Vec<ModelScore>>,
    pub converged: bool,
    pub attractor: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
pub struct OrbitIteration {
    pub step: usize,
    pub image_prompt: String,
    pub image_path: String,
    pub seed: u64,
    pub vision_description: String,
    pub extracted_concepts: Vec<String>,
    pub emoji_encoding: String,
    pub semantic_vector: Vec<f64>,
    pub lattice_scores: HashMap<String, f64>,
}

#[derive(Debug, Serialize, Clone)]
pub struct SemanticIndex {
    pub concepts: HashMap<String, ConceptEntry>,
    pub emoji_patterns: HashMap<String, usize>,
    pub convergence_graph: Vec<(usize, f64)>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ConceptEntry {
    pub concept: String,
    pub frequency: usize,
    pub first_appearance: usize,
    pub emoji: String,
    pub related_concepts: Vec<String>,
}

pub struct ProgressivePipeline {
    flux_gen: FluxGenerator,
    llava: LlavaAnalyzer,
    output_dir: String,
    model_scorer: ModelScorer,
}

impl ProgressivePipeline {
    pub fn new(output_dir: &str) -> Result<Self> {
        fs::create_dir_all(output_dir).ok();
        fs::create_dir_all(format!("{}/images", output_dir)).ok();
        fs::create_dir_all(format!("{}/orbits", output_dir)).ok();
        
        Ok(Self {
            flux_gen: FluxGenerator::new()?,
            llava: LlavaAnalyzer::new()?,
            output_dir: output_dir.to_string(),
            model_scorer: ModelScorer::new(),
        })
    }
    
    /// Step 1: Generate image with FLUX.1-dev
    pub async fn generate_image(&self, prompt: &str, seed: u64, step: usize) -> Result<String> {
        println!("  🎨 Generating image (step {})...", step);
        
        let image_bytes = self.flux_gen.generate(prompt, seed).await?;
        
        let image_path = format!("{}/images/step_{:03}_seed_{}.png", 
                                self.output_dir, step, seed);
        
        fs::write(&image_path, image_bytes)?;
        
        println!("     ✓ Saved: {}", image_path);
        Ok(image_path)
    }
    
    /// Step 2: Score with entire model lattice
    pub async fn score_with_lattice(&mut self, image_path: &str, ground_truth: &[String]) -> Result<HashMap<String, f64>> {
        println!("  📊 Scoring with model lattice...");
        
        self.model_scorer.score_all(image_path, ground_truth).await?;
        
        let mut scores = HashMap::new();
        for (model_id, score) in &self.model_scorer.get_lattice().scores {
            scores.insert(model_id.clone(), score.accuracy);
        }
        
        Ok(scores)
    }
    
    /// Step 3: Vision model analyzes image with LLaVA
    pub async fn analyze_with_vision(&self, image_path: &str) -> Result<String> {
        println!("  👁️  Analyzing with LLaVA...");
        
        let image_bytes = fs::read(image_path)?;
        let description = self.llava.analyze(
            &image_bytes,
            "Describe everything you see in this image, including all text, objects, and the scene."
        ).await?;
        
        println!("     ✓ Analysis complete");
        Ok(description)
    }
    
    /// Step 4: Extract concepts and encode
    pub fn extract_and_encode(&self, description: &str) -> (Vec<String>, String, Vec<f64>) {
        println!("  🔍 Extracting concepts...");
        
        let concepts = self.extract_concepts(description);
        let emoji = self.encode_as_emoji(&concepts);
        let vector = self.compute_semantic_vector(&concepts);
        
        println!("     Concepts: {:?}", concepts);
        println!("     Emoji: {}", emoji);
        
        (concepts, emoji, vector)
    }
    
    /// Step 5: Generate next prompt from vision output (feedback)
    pub fn generate_next_prompt(&self, description: &str, emoji: &str) -> String {
        format!("reflect on: {} {}", emoji, description)
    }
    
    /// Run full automorphic orbit with lattice scoring
    pub async fn run_orbit(&mut self, initial_prompt: &str, initial_seed: u64, max_iterations: usize) -> Result<AutomorphicOrbit> {
        println!("\n🔄 Starting Automorphic Orbit with Model Lattice");
        println!("=================================================");
        println!("Initial prompt: {}", initial_prompt);
        println!("Initial seed: {}", initial_seed);
        println!("Max iterations: {}\n", max_iterations);
        
        let mut orbit = AutomorphicOrbit {
            orbit_id: format!("orbit_{}", initial_seed),
            initial_seed,
            iterations: Vec::new(),
            semantic_index: SemanticIndex {
                concepts: HashMap::new(),
                emoji_patterns: HashMap::new(),
                convergence_graph: Vec::new(),
            },
            model_scores: HashMap::new(),
            converged: false,
            attractor: None,
        };
        
        let mut current_prompt = initial_prompt.to_string();
        let mut seed = initial_seed;
        
        for step in 0..max_iterations {
            println!("\n--- Iteration {} ---", step);
            
            // 1. Generate image
            let image_path = self.generate_image(&current_prompt, seed, step).await?;
            
            // 2. Vision analysis
            let description = self.analyze_with_vision(&image_path).await?;
            
            // 3. Extract & encode
            let (concepts, emoji, vector) = self.extract_and_encode(&description);
            
            // 4. Score with lattice
            let lattice_scores = self.score_with_lattice(&image_path, &concepts).await?;
            
            // 5. Create iteration record
            let iteration = OrbitIteration {
                step,
                image_prompt: current_prompt.clone(),
                image_path: image_path.clone(),
                seed,
                vision_description: description.clone(),
                extracted_concepts: concepts.clone(),
                emoji_encoding: emoji.clone(),
                semantic_vector: vector.clone(),
                lattice_scores: lattice_scores.clone(),
            };
            
            // 6. Update semantic index
            self.update_semantic_index(&mut orbit.semantic_index, &iteration);
            
            // 7. Check convergence
            if step > 0 {
                let similarity = self.compute_similarity(
                    &orbit.iterations[step-1].semantic_vector,
                    &vector
                );
                orbit.semantic_index.convergence_graph.push((step, similarity));
                
                println!("  📊 Similarity to previous: {:.2}%", similarity * 100.0);
                
                if similarity > 0.95 {
                    println!("\n  ✓ CONVERGED at iteration {}", step);
                    orbit.converged = true;
                    orbit.attractor = Some(emoji.clone());
                    orbit.iterations.push(iteration);
                    break;
                }
            }
            
            orbit.iterations.push(iteration);
            
            // 8. Generate next prompt (automorphic feedback)
            current_prompt = self.generate_next_prompt(&description, &emoji);
            seed = seed.wrapping_add(1);
            
            // Progressive save
            self.save_orbit_progress(&orbit)?;
        }
        
        // Save lattice report
        let lattice_report = self.model_scorer.generate_report();
        fs::write(
            format!("{}/orbits/{}_LATTICE.md", self.output_dir, orbit.orbit_id),
            lattice_report
        )?;
        
        // Final save
        self.save_orbit_final(&orbit)?;
        
        Ok(orbit)
    }
    
    fn extract_concepts(&self, text: &str) -> Vec<String> {
        let keywords = ["life", "tree", "water", "train", "text", "symmetry", 
                       "self", "awareness", "consciousness", "monster", "group"];
        
        let mut concepts = Vec::new();
        let lower = text.to_lowercase();
        
        for keyword in keywords {
            if lower.contains(keyword) {
                concepts.push(keyword.to_string());
            }
        }
        
        concepts
    }
    
    fn encode_as_emoji(&self, concepts: &[String]) -> String {
        let mut emoji = String::new();
        
        for concept in concepts {
            let e = match concept.as_str() {
                "life" => "🌱",
                "tree" => "🌳",
                "water" => "🌊",
                "train" => "🚂",
                "text" => "📝",
                "symmetry" => "🔄",
                "self" => "👁️",
                "awareness" => "💡",
                "consciousness" => "🧠",
                "monster" => "🎪",
                "group" => "🔢",
                _ => "❓",
            };
            emoji.push_str(e);
        }
        
        if emoji.is_empty() { emoji.push_str("❓"); }
        emoji
    }
    
    fn compute_semantic_vector(&self, concepts: &[String]) -> Vec<f64> {
        let mut vec = vec![0.0; 20];
        
        for (i, concept) in concepts.iter().enumerate() {
            if i < vec.len() {
                vec[i] = 1.0;
            }
        }
        
        let mag: f64 = vec.iter().map(|x| x*x).sum::<f64>().sqrt();
        if mag > 0.0 {
            for x in &mut vec {
                *x /= mag;
            }
        }
        
        vec
    }
    
    fn compute_similarity(&self, v1: &[f64], v2: &[f64]) -> f64 {
        let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        dot.max(0.0).min(1.0)
    }
    
    fn update_semantic_index(&self, index: &mut SemanticIndex, iteration: &OrbitIteration) {
        for concept in &iteration.extracted_concepts {
            index.concepts.entry(concept.clone())
                .and_modify(|e| e.frequency += 1)
                .or_insert(ConceptEntry {
                    concept: concept.clone(),
                    frequency: 1,
                    first_appearance: iteration.step,
                    emoji: self.encode_as_emoji(&[concept.clone()]),
                    related_concepts: Vec::new(),
                });
        }
        
        *index.emoji_patterns.entry(iteration.emoji_encoding.clone())
            .or_insert(0) += 1;
    }
    
    fn save_orbit_progress(&self, orbit: &AutomorphicOrbit) -> Result<()> {
        let path = format!("{}/orbits/{}_progress.json", 
                          self.output_dir, orbit.orbit_id);
        fs::write(path, serde_json::to_string_pretty(orbit)?)?;
        Ok(())
    }
    
    fn save_orbit_final(&self, orbit: &AutomorphicOrbit) -> Result<()> {
        let json_path = format!("{}/orbits/{}.json", 
                               self.output_dir, orbit.orbit_id);
        fs::write(json_path, serde_json::to_string_pretty(orbit)?)?;
        
        let report = self.generate_orbit_report(orbit);
        let report_path = format!("{}/orbits/{}_REPORT.md", 
                                 self.output_dir, orbit.orbit_id);
        fs::write(report_path, report)?;
        
        Ok(())
    }
    
    fn generate_orbit_report(&self, orbit: &AutomorphicOrbit) -> String {
        let mut report = String::new();
        
        report.push_str(&format!("# Automorphic Orbit: {}\n\n", orbit.orbit_id));
        report.push_str(&format!("**Initial Seed:** {}\n", orbit.initial_seed));
        report.push_str(&format!("**Iterations:** {}\n", orbit.iterations.len()));
        report.push_str(&format!("**Converged:** {}\n", orbit.converged));
        
        if let Some(attractor) = &orbit.attractor {
            report.push_str(&format!("**Attractor:** {}\n", attractor));
        }
        
        report.push_str("\n## Model Lattice Scores\n\n");
        report.push_str("Average scores across all iterations:\n\n");
        
        // Aggregate scores
        let mut avg_scores: HashMap<String, Vec<f64>> = HashMap::new();
        for iter in &orbit.iterations {
            for (model, score) in &iter.lattice_scores {
                avg_scores.entry(model.clone()).or_insert_with(Vec::new).push(*score);
            }
        }
        
        for (model, scores) in avg_scores {
            let avg = scores.iter().sum::<f64>() / scores.len() as f64;
            report.push_str(&format!("- {}: {:.1}%\n", model, avg * 100.0));
        }
        
        report.push_str("\n## Semantic Index\n\n");
        report.push_str(&format!("**Unique Concepts:** {}\n", orbit.semantic_index.concepts.len()));
        report.push_str(&format!("**Emoji Patterns:** {}\n\n", orbit.semantic_index.emoji_patterns.len()));
        
        report.push_str("### Top Concepts\n\n");
        let mut concepts: Vec<_> = orbit.semantic_index.concepts.values().collect();
        concepts.sort_by_key(|c| std::cmp::Reverse(c.frequency));
        
        for concept in concepts.iter().take(10) {
            report.push_str(&format!("- {} **{}** (appeared {} times, first at step {})\n",
                                    concept.emoji, concept.concept, 
                                    concept.frequency, concept.first_appearance));
        }
        
        report.push_str("\n### Emoji Timeline\n\n");
        for iter in &orbit.iterations {
            report.push_str(&format!("**Step {}:** {} → {}\n", 
                                    iter.step, iter.emoji_encoding, 
                                    iter.extracted_concepts.join(", ")));
        }
        
        report
    }
}


#[derive(Debug, Serialize, Clone)]
pub struct OrbitIteration {
    pub step: usize,
    pub image_prompt: String,
    pub image_path: String,
    pub seed: u64,
    pub vision_description: String,
    pub extracted_concepts: Vec<String>,
    pub emoji_encoding: String,
    pub semantic_vector: Vec<f64>,
}

#[derive(Debug, Serialize, Clone)]
pub struct SemanticIndex {
    pub concepts: HashMap<String, ConceptEntry>,
    pub emoji_patterns: HashMap<String, usize>,
    pub convergence_graph: Vec<(usize, f64)>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ConceptEntry {
    pub concept: String,
    pub frequency: usize,
    pub first_appearance: usize,
    pub emoji: String,
    pub related_concepts: Vec<String>,
}

pub struct ProgressivePipeline {
    image_model: String,
    vision_model: String,
    output_dir: String,
}

impl ProgressivePipeline {
    pub fn new(output_dir: &str) -> Self {
        fs::create_dir_all(output_dir).ok();
        fs::create_dir_all(format!("{}/images", output_dir)).ok();
        fs::create_dir_all(format!("{}/orbits", output_dir)).ok();
        
        Self {
            image_model: "FLUX.1-dev".to_string(),
            vision_model: "llava".to_string(),
            output_dir: output_dir.to_string(),
        }
    }
    
    /// Step 1: Generate image with seed
    pub async fn generate_image(&self, prompt: &str, seed: u64, step: usize) -> Result<String> {
        println!("  🎨 Generating image (step {})...", step);
        println!("     Model: {}", self.image_model);
        println!("     Seed: {}", seed);
        println!("     Prompt: {}", prompt);
        
        let image_path = format!("{}/images/step_{:03}_seed_{}.png", 
                                self.output_dir, step, seed);
        
        // TODO: Call actual FLUX.1-dev via mistral.rs or HF API
        // For now, create placeholder
        fs::write(&image_path, b"[image data]")?;
        
        println!("     ✓ Saved: {}", image_path);
        Ok(image_path)
    }
    
    /// Step 2: Vision model analyzes image
    pub async fn analyze_with_vision(&self, image_path: &str) -> Result<String> {
        println!("  👁️  Analyzing with vision model...");
        println!("     Model: {}", self.vision_model);
        
        // TODO: Call actual LLaVA via mistral.rs
        // Prompt: "Describe everything you see in this image, including all text"
        
        let description = format!(
            "[Vision analysis of {}]\nDescribe: text, objects, scene, composition",
            image_path
        );
        
        println!("     ✓ Analysis complete");
        Ok(description)
    }
    
    /// Step 3: Extract concepts and encode
    pub fn extract_and_encode(&self, description: &str) -> (Vec<String>, String, Vec<f64>) {
        println!("  🔍 Extracting concepts...");
        
        let concepts = self.extract_concepts(description);
        let emoji = self.encode_as_emoji(&concepts);
        let vector = self.compute_semantic_vector(&concepts);
        
        println!("     Concepts: {:?}", concepts);
        println!("     Emoji: {}", emoji);
        
        (concepts, emoji, vector)
    }
    
    /// Step 4: Generate next prompt from vision output (feedback)
    pub fn generate_next_prompt(&self, description: &str, emoji: &str) -> String {
        // Feed vision output back as next image prompt
        format!("reflect on: {} {}", emoji, description)
    }
    
    /// Run full automorphic orbit
    pub async fn run_orbit(&mut self, initial_prompt: &str, initial_seed: u64, max_iterations: usize) -> Result<AutomorphicOrbit> {
        println!("\n🔄 Starting Automorphic Orbit");
        println!("================================");
        println!("Initial prompt: {}", initial_prompt);
        println!("Initial seed: {}", initial_seed);
        println!("Max iterations: {}\n", max_iterations);
        
        let mut orbit = AutomorphicOrbit {
            orbit_id: format!("orbit_{}", initial_seed),
            initial_seed,
            iterations: Vec::new(),
            semantic_index: SemanticIndex {
                concepts: HashMap::new(),
                emoji_patterns: HashMap::new(),
                convergence_graph: Vec::new(),
            },
            converged: false,
            attractor: None,
        };
        
        let mut current_prompt = initial_prompt.to_string();
        let mut seed = initial_seed;
        
        for step in 0..max_iterations {
            println!("\n--- Iteration {} ---", step);
            
            // 1. Generate image
            let image_path = self.generate_image(&current_prompt, seed, step).await?;
            
            // 2. Vision analysis
            let description = self.analyze_with_vision(&image_path).await?;
            
            // 3. Extract & encode
            let (concepts, emoji, vector) = self.extract_and_encode(&description);
            
            // 4. Create iteration record
            let iteration = OrbitIteration {
                step,
                image_prompt: current_prompt.clone(),
                image_path: image_path.clone(),
                seed,
                vision_description: description.clone(),
                extracted_concepts: concepts.clone(),
                emoji_encoding: emoji.clone(),
                semantic_vector: vector.clone(),
            };
            
            // 5. Update semantic index
            self.update_semantic_index(&mut orbit.semantic_index, &iteration);
            
            // 6. Check convergence
            if step > 0 {
                let similarity = self.compute_similarity(
                    &orbit.iterations[step-1].semantic_vector,
                    &vector
                );
                orbit.semantic_index.convergence_graph.push((step, similarity));
                
                println!("  📊 Similarity to previous: {:.2}%", similarity * 100.0);
                
                if similarity > 0.95 {
                    println!("\n  ✓ CONVERGED at iteration {}", step);
                    orbit.converged = true;
                    orbit.attractor = Some(emoji.clone());
                    orbit.iterations.push(iteration);
                    break;
                }
            }
            
            orbit.iterations.push(iteration);
            
            // 7. Generate next prompt (automorphic feedback)
            current_prompt = self.generate_next_prompt(&description, &emoji);
            seed = seed.wrapping_add(1);
            
            // Progressive save
            self.save_orbit_progress(&orbit)?;
        }
        
        // Final save
        self.save_orbit_final(&orbit)?;
        
        Ok(orbit)
    }
    
    fn extract_concepts(&self, text: &str) -> Vec<String> {
        let keywords = ["life", "tree", "water", "train", "text", "symmetry", 
                       "self", "awareness", "consciousness", "monster", "group"];
        
        let mut concepts = Vec::new();
        let lower = text.to_lowercase();
        
        for keyword in keywords {
            if lower.contains(keyword) {
                concepts.push(keyword.to_string());
            }
        }
        
        concepts
    }
    
    fn encode_as_emoji(&self, concepts: &[String]) -> String {
        let mut emoji = String::new();
        
        for concept in concepts {
            let e = match concept.as_str() {
                "life" => "🌱",
                "tree" => "🌳",
                "water" => "🌊",
                "train" => "🚂",
                "text" => "📝",
                "symmetry" => "🔄",
                "self" => "👁️",
                "awareness" => "💡",
                "consciousness" => "🧠",
                "monster" => "🎪",
                "group" => "🔢",
                _ => "❓",
            };
            emoji.push_str(e);
        }
        
        if emoji.is_empty() { emoji.push_str("❓"); }
        emoji
    }
    
    fn compute_semantic_vector(&self, concepts: &[String]) -> Vec<f64> {
        // Simple bag-of-words vector
        let mut vec = vec![0.0; 20];
        
        for (i, concept) in concepts.iter().enumerate() {
            if i < vec.len() {
                vec[i] = 1.0;
            }
        }
        
        // Normalize
        let mag: f64 = vec.iter().map(|x| x*x).sum::<f64>().sqrt();
        if mag > 0.0 {
            for x in &mut vec {
                *x /= mag;
            }
        }
        
        vec
    }
    
    fn compute_similarity(&self, v1: &[f64], v2: &[f64]) -> f64 {
        let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        dot.max(0.0).min(1.0)
    }
    
    fn update_semantic_index(&self, index: &mut SemanticIndex, iteration: &OrbitIteration) {
        // Update concept frequencies
        for concept in &iteration.extracted_concepts {
            index.concepts.entry(concept.clone())
                .and_modify(|e| e.frequency += 1)
                .or_insert(ConceptEntry {
                    concept: concept.clone(),
                    frequency: 1,
                    first_appearance: iteration.step,
                    emoji: self.encode_as_emoji(&[concept.clone()]),
                    related_concepts: Vec::new(),
                });
        }
        
        // Update emoji pattern frequencies
        *index.emoji_patterns.entry(iteration.emoji_encoding.clone())
            .or_insert(0) += 1;
    }
    
    fn save_orbit_progress(&self, orbit: &AutomorphicOrbit) -> Result<()> {
        let path = format!("{}/orbits/{}_progress.json", 
                          self.output_dir, orbit.orbit_id);
        fs::write(path, serde_json::to_string_pretty(orbit)?)?;
        Ok(())
    }
    
    fn save_orbit_final(&self, orbit: &AutomorphicOrbit) -> Result<()> {
        // Save JSON
        let json_path = format!("{}/orbits/{}.json", 
                               self.output_dir, orbit.orbit_id);
        fs::write(json_path, serde_json::to_string_pretty(orbit)?)?;
        
        // Save report
        let report = self.generate_orbit_report(orbit);
        let report_path = format!("{}/orbits/{}_REPORT.md", 
                                 self.output_dir, orbit.orbit_id);
        fs::write(report_path, report)?;
        
        Ok(())
    }
    
    fn generate_orbit_report(&self, orbit: &AutomorphicOrbit) -> String {
        let mut report = String::new();
        
        report.push_str(&format!("# Automorphic Orbit: {}\n\n", orbit.orbit_id));
        report.push_str(&format!("**Initial Seed:** {}\n", orbit.initial_seed));
        report.push_str(&format!("**Iterations:** {}\n", orbit.iterations.len()));
        report.push_str(&format!("**Converged:** {}\n", orbit.converged));
        
        if let Some(attractor) = &orbit.attractor {
            report.push_str(&format!("**Attractor:** {}\n", attractor));
        }
        
        report.push_str("\n## Semantic Index\n\n");
        report.push_str(&format!("**Unique Concepts:** {}\n", orbit.semantic_index.concepts.len()));
        report.push_str(&format!("**Emoji Patterns:** {}\n\n", orbit.semantic_index.emoji_patterns.len()));
        
        report.push_str("### Top Concepts\n\n");
        let mut concepts: Vec<_> = orbit.semantic_index.concepts.values().collect();
        concepts.sort_by_key(|c| std::cmp::Reverse(c.frequency));
        
        for concept in concepts.iter().take(10) {
            report.push_str(&format!("- {} **{}** (appeared {} times, first at step {})\n",
                                    concept.emoji, concept.concept, 
                                    concept.frequency, concept.first_appearance));
        }
        
        report.push_str("\n### Emoji Timeline\n\n");
        for iter in &orbit.iterations {
            report.push_str(&format!("**Step {}:** {} → {}\n", 
                                    iter.step, iter.emoji_encoding, 
                                    iter.extracted_concepts.join(", ")));
        }
        
        report.push_str("\n### Convergence Graph\n\n");
        for (step, similarity) in &orbit.semantic_index.convergence_graph {
            let bar = "█".repeat((similarity * 50.0) as usize);
            report.push_str(&format!("Step {}: {} {:.2}%\n", step, bar, similarity * 100.0));
        }
        
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_pipeline() {
        let mut pipeline = ProgressivePipeline::new("test_output");
        let orbit = pipeline.run_orbit("unconstrained", 2437596016, 3).await.unwrap();
        assert!(orbit.iterations.len() > 0);
    }
}
