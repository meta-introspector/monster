use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model lattice - hierarchy of models from simple to complex
#[derive(Debug, Serialize, Clone)]
pub struct ModelLattice {
    pub levels: Vec<LatticeLevel>,
    pub scores: HashMap<String, ModelScore>,
}

#[derive(Debug, Serialize, Clone)]
pub struct LatticeLevel {
    pub level: usize,
    pub name: String,
    pub models: Vec<ModelNode>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ModelNode {
    pub model_id: String,
    pub model_type: ModelType,
    pub complexity: f64,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Serialize, Clone)]
pub enum ModelType {
    OpenCV,           // Level 0: Basic CV
    ClassicalML,      // Level 1: Traditional ML
    SmallVision,      // Level 2: Small vision models
    LargeVision,      // Level 3: Large vision models
    MultiModal,       // Level 4: Multi-modal
}

#[derive(Debug, Serialize, Clone)]
pub struct ModelScore {
    pub model_id: String,
    pub accuracy: f64,
    pub latency_ms: u64,
    pub concepts_detected: usize,
    pub emoji_match: f64,
    pub convergence_contribution: f64,
}

pub struct ModelScorer {
    lattice: ModelLattice,
}

impl ModelScorer {
    pub fn new() -> Self {
        let lattice = Self::build_lattice();
        Self { lattice }
    }
    
    fn build_lattice() -> ModelLattice {
        ModelLattice {
            levels: vec![
                // Level 0: OpenCV - basic computer vision
                LatticeLevel {
                    level: 0,
                    name: "Classical CV".to_string(),
                    models: vec![
                        ModelNode {
                            model_id: "opencv-edge".to_string(),
                            model_type: ModelType::OpenCV,
                            complexity: 0.1,
                            capabilities: vec!["edges".to_string(), "contours".to_string()],
                        },
                        ModelNode {
                            model_id: "opencv-text".to_string(),
                            model_type: ModelType::OpenCV,
                            complexity: 0.2,
                            capabilities: vec!["text".to_string(), "ocr".to_string()],
                        },
                        ModelNode {
                            model_id: "opencv-color".to_string(),
                            model_type: ModelType::OpenCV,
                            complexity: 0.1,
                            capabilities: vec!["colors".to_string(), "histograms".to_string()],
                        },
                    ],
                },
                
                // Level 1: Classical ML
                LatticeLevel {
                    level: 1,
                    name: "Classical ML".to_string(),
                    models: vec![
                        ModelNode {
                            model_id: "sift-features".to_string(),
                            model_type: ModelType::ClassicalML,
                            complexity: 0.3,
                            capabilities: vec!["features".to_string(), "keypoints".to_string()],
                        },
                        ModelNode {
                            model_id: "hog-detector".to_string(),
                            model_type: ModelType::ClassicalML,
                            complexity: 0.4,
                            capabilities: vec!["objects".to_string(), "shapes".to_string()],
                        },
                    ],
                },
                
                // Level 2: Small vision models
                LatticeLevel {
                    level: 2,
                    name: "Small Vision".to_string(),
                    models: vec![
                        ModelNode {
                            model_id: "mobilenet".to_string(),
                            model_type: ModelType::SmallVision,
                            complexity: 0.5,
                            capabilities: vec!["classification".to_string(), "objects".to_string()],
                        },
                        ModelNode {
                            model_id: "clip-small".to_string(),
                            model_type: ModelType::SmallVision,
                            complexity: 0.6,
                            capabilities: vec!["vision-language".to_string(), "embeddings".to_string()],
                        },
                    ],
                },
                
                // Level 3: Large vision models
                LatticeLevel {
                    level: 3,
                    name: "Large Vision".to_string(),
                    models: vec![
                        ModelNode {
                            model_id: "llava-7b".to_string(),
                            model_type: ModelType::LargeVision,
                            complexity: 0.8,
                            capabilities: vec!["description".to_string(), "reasoning".to_string()],
                        },
                        ModelNode {
                            model_id: "bakllava".to_string(),
                            model_type: ModelType::LargeVision,
                            complexity: 0.85,
                            capabilities: vec!["diagrams".to_string(), "math".to_string()],
                        },
                    ],
                },
                
                // Level 4: Multi-modal
                LatticeLevel {
                    level: 4,
                    name: "Multi-Modal".to_string(),
                    models: vec![
                        ModelNode {
                            model_id: "gpt4-vision".to_string(),
                            model_type: ModelType::MultiModal,
                            complexity: 1.0,
                            capabilities: vec!["full-understanding".to_string(), "reasoning".to_string()],
                        },
                    ],
                },
            ],
            scores: HashMap::new(),
        }
    }
    
    /// Score image with all models in lattice
    pub async fn score_all(&mut self, image_path: &str, ground_truth: &[String]) -> Result<()> {
        println!("📊 Scoring with model lattice...");
        
        for level in &self.lattice.levels {
            println!("\n  Level {}: {}", level.level, level.name);
            
            for model in &level.models {
                let score = self.score_model(&model, image_path, ground_truth).await?;
                println!("    {} - accuracy: {:.2}%, latency: {}ms", 
                         model.model_id, score.accuracy * 100.0, score.latency_ms);
                
                self.lattice.scores.insert(model.model_id.clone(), score);
            }
        }
        
        Ok(())
    }
    
    async fn score_model(&self, model: &ModelNode, image_path: &str, ground_truth: &[String]) -> Result<ModelScore> {
        let start = std::time::Instant::now();
        
        let detected = match model.model_type {
            ModelType::OpenCV => self.run_opencv(&model.model_id, image_path)?,
            ModelType::ClassicalML => self.run_classical_ml(&model.model_id, image_path)?,
            ModelType::SmallVision => self.run_small_vision(&model.model_id, image_path).await?,
            ModelType::LargeVision => self.run_large_vision(&model.model_id, image_path).await?,
            ModelType::MultiModal => self.run_multimodal(&model.model_id, image_path).await?,
        };
        
        let latency = start.elapsed().as_millis() as u64;
        
        // Calculate accuracy
        let matches = detected.iter()
            .filter(|d| ground_truth.iter().any(|g| g.contains(*d)))
            .count();
        let accuracy = matches as f64 / ground_truth.len().max(1) as f64;
        
        Ok(ModelScore {
            model_id: model.model_id.clone(),
            accuracy,
            latency_ms: latency,
            concepts_detected: detected.len(),
            emoji_match: 0.0, // TODO: compute
            convergence_contribution: accuracy * (1.0 - model.complexity),
        })
    }
    
    fn run_opencv(&self, model_id: &str, _image_path: &str) -> Result<Vec<String>> {
        // TODO: Actual OpenCV calls
        let detected = match model_id {
            "opencv-edge" => vec!["edges".to_string(), "lines".to_string()],
            "opencv-text" => vec!["text".to_string(), "characters".to_string()],
            "opencv-color" => vec!["colors".to_string(), "regions".to_string()],
            _ => vec![],
        };
        Ok(detected)
    }
    
    fn run_classical_ml(&self, model_id: &str, _image_path: &str) -> Result<Vec<String>> {
        let detected = match model_id {
            "sift-features" => vec!["keypoints".to_string(), "features".to_string()],
            "hog-detector" => vec!["objects".to_string(), "shapes".to_string()],
            _ => vec![],
        };
        Ok(detected)
    }
    
    async fn run_small_vision(&self, model_id: &str, _image_path: &str) -> Result<Vec<String>> {
        let detected = match model_id {
            "mobilenet" => vec!["tree".to_string(), "water".to_string()],
            "clip-small" => vec!["nature".to_string(), "scene".to_string()],
            _ => vec![],
        };
        Ok(detected)
    }
    
    async fn run_large_vision(&self, model_id: &str, _image_path: &str) -> Result<Vec<String>> {
        // TODO: Actual LLaVA inference via mistral.rs
        let detected = match model_id {
            "llava-7b" => vec!["tree".to_string(), "text".to_string(), "life".to_string()],
            "bakllava" => vec!["diagram".to_string(), "symmetry".to_string()],
            _ => vec![],
        };
        Ok(detected)
    }
    
    async fn run_multimodal(&self, _model_id: &str, _image_path: &str) -> Result<Vec<String>> {
        Ok(vec!["comprehensive".to_string(), "understanding".to_string()])
    }
    
    /// Generate lattice report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str("# Model Lattice Scoring Report\n\n");
        
        // Summary table
        report.push_str("## Summary\n\n");
        report.push_str("| Model | Level | Accuracy | Latency | Concepts | Score |\n");
        report.push_str("|-------|-------|----------|---------|----------|-------|\n");
        
        let mut scores: Vec<_> = self.lattice.scores.values().collect();
        scores.sort_by(|a, b| b.convergence_contribution.partial_cmp(&a.convergence_contribution).unwrap());
        
        for score in &scores {
            let level = self.get_model_level(&score.model_id);
            report.push_str(&format!(
                "| {} | {} | {:.1}% | {}ms | {} | {:.3} |\n",
                score.model_id, level, score.accuracy * 100.0, 
                score.latency_ms, score.concepts_detected, score.convergence_contribution
            ));
        }
        
        // Level analysis
        report.push_str("\n## Analysis by Level\n\n");
        
        for level in &self.lattice.levels {
            report.push_str(&format!("### Level {}: {}\n\n", level.level, level.name));
            
            for model in &level.models {
                if let Some(score) = self.lattice.scores.get(&model.model_id) {
                    report.push_str(&format!("**{}**\n", model.model_id));
                    report.push_str(&format!("- Complexity: {:.1}\n", model.complexity));
                    report.push_str(&format!("- Accuracy: {:.1}%\n", score.accuracy * 100.0));
                    report.push_str(&format!("- Latency: {}ms\n", score.latency_ms));
                    report.push_str(&format!("- Capabilities: {}\n\n", model.capabilities.join(", ")));
                }
            }
        }
        
        // Insights
        report.push_str("\n## Insights\n\n");
        report.push_str("- Simple models (OpenCV) are fast but limited\n");
        report.push_str("- Classical ML provides good feature detection\n");
        report.push_str("- Small vision models balance speed and accuracy\n");
        report.push_str("- Large vision models provide deep understanding\n");
        report.push_str("- Multi-modal models are most comprehensive\n");
        
        report
    }
    
    fn get_model_level(&self, model_id: &str) -> usize {
        for level in &self.lattice.levels {
            if level.models.iter().any(|m| m.model_id == model_id) {
                return level.level;
            }
        }
        0
    }
    
    pub fn get_lattice(&self) -> &ModelLattice {
        &self.lattice
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_lattice() {
        let mut scorer = ModelScorer::new();
        assert_eq!(scorer.lattice.levels.len(), 5);
        
        let ground_truth = vec!["tree".to_string(), "text".to_string()];
        scorer.score_all("test.png", &ground_truth).await.unwrap();
        
        assert!(scorer.lattice.scores.len() > 0);
    }
}
