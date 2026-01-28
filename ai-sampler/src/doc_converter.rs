use anyhow::Result;
use std::process::Command;
use std::fs;

/// Convert code and docs to PDF images for vision models
pub struct DocConverter {
    output_dir: String,
}

impl DocConverter {
    pub fn new(output_dir: &str) -> Self {
        fs::create_dir_all(output_dir).ok();
        Self {
            output_dir: output_dir.to_string(),
        }
    }
    
    /// Convert Rust source to PDF
    pub fn rust_to_pdf(&self, source: &str, output: &str) -> Result<()> {
        // Use enscript + ps2pdf
        let ps = format!("{}.ps", output);
        
        Command::new("enscript")
            .args(&[
                "-E", "rust",
                "--color",
                "-p", &ps,
                source
            ])
            .output()?;
        
        Command::new("ps2pdf")
            .args(&[&ps, &format!("{}/{}.pdf", self.output_dir, output)])
            .output()?;
        
        fs::remove_file(ps)?;
        Ok(())
    }
    
    /// Convert Lean proof to PDF
    pub fn lean_to_pdf(&self, source: &str, output: &str) -> Result<()> {
        let ps = format!("{}.ps", output);
        
        Command::new("enscript")
            .args(&[
                "-E", "haskell", // Close enough for Lean
                "--color",
                "-p", &ps,
                source
            ])
            .output()?;
        
        Command::new("ps2pdf")
            .args(&[&ps, &format!("{}/{}.pdf", self.output_dir, output)])
            .output()?;
        
        fs::remove_file(ps)?;
        Ok(())
    }
    
    /// Convert markdown to PDF
    pub fn md_to_pdf(&self, source: &str, output: &str) -> Result<()> {
        Command::new("pandoc")
            .args(&[
                source,
                "-o", &format!("{}/{}.pdf", self.output_dir, output),
                "--pdf-engine=xelatex"
            ])
            .output()?;
        Ok(())
    }
    
    /// Convert PDF to images for vision models
    pub fn pdf_to_images(&self, pdf: &str, prefix: &str) -> Result<Vec<String>> {
        let output_pattern = format!("{}/{}_%03d.png", self.output_dir, prefix);
        
        Command::new("pdftoppm")
            .args(&[
                "-png",
                "-r", "150", // 150 DPI
                pdf,
                &format!("{}/{}", self.output_dir, prefix)
            ])
            .output()?;
        
        // Collect generated images
        let mut images = Vec::new();
        for entry in fs::read_dir(&self.output_dir)? {
            let entry = entry?;
            let path = entry.path();
            if let Some(name) = path.file_name() {
                if name.to_string_lossy().starts_with(prefix) {
                    images.push(path.to_string_lossy().to_string());
                }
            }
        }
        
        Ok(images)
    }
    
    /// Convert all project files
    pub fn convert_all(&self) -> Result<Vec<String>> {
        let mut all_images = Vec::new();
        
        println!("📄 Converting Rust sources...");
        self.rust_to_pdf("../src/main.rs", "main")?;
        self.rust_to_pdf("../src/all_groups.rs", "all_groups")?;
        
        println!("📐 Converting Lean proofs...");
        self.lean_to_pdf("../MonsterLean/MonsterWalk.lean", "monster_walk_proof")?;
        self.lean_to_pdf("../MonsterLean/BottPeriodicity.lean", "bott_proof")?;
        
        println!("📝 Converting documentation...");
        self.md_to_pdf("../README.md", "readme")?;
        self.md_to_pdf("../BOTT_PERIODICITY.md", "bott_doc")?;
        
        println!("🖼️  Converting PDFs to images...");
        all_images.extend(self.pdf_to_images(&format!("{}/main.pdf", self.output_dir), "main")?);
        all_images.extend(self.pdf_to_images(&format!("{}/monster_walk_proof.pdf", self.output_dir), "proof")?);
        all_images.extend(self.pdf_to_images(&format!("{}/readme.pdf", self.output_dir), "doc")?);
        
        println!("✓ Generated {} images", all_images.len());
        
        Ok(all_images)
    }
}
