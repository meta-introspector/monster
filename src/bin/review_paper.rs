use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use serde::{Deserialize, Serialize};
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Persona {
    focus: String,
    prompt: String,
}

#[derive(Debug, Serialize)]
struct Review {
    page: usize,
    persona: String,
    role: String,
    content: String,
}

fn scholars() -> HashMap<String, Persona> {
    let mut map = HashMap::new();
    
    map.insert("mathematician".to_string(), Persona {
        focus: "Mathematical rigor, proof correctness, notation consistency".to_string(),
        prompt: "You are a pure mathematician. Review this page for: 1) Proof correctness 2) Notation consistency 3) Missing lemmas 4) Logical gaps. Be rigorous and precise.".to_string(),
    });
    
    map.insert("computer_scientist".to_string(), Persona {
        focus: "Algorithmic complexity, implementation feasibility, data structures".to_string(),
        prompt: "You are a computer scientist. Review for: 1) Algorithm correctness 2) Complexity analysis 3) Implementation issues 4) Data structure choices. Be practical.".to_string(),
    });
    
    map.insert("group_theorist".to_string(), Persona {
        focus: "Group theory correctness, Monster group properties, representation theory".to_string(),
        prompt: "You are a group theorist specializing in sporadic groups. Review for: 1) Monster group properties 2) Representation accuracy 3) Modular forms 4) J-invariant usage.".to_string(),
    });
    
    map
}

fn muses() -> HashMap<String, Persona> {
    let mut map = HashMap::new();
    
    map.insert("visionary".to_string(), Persona {
        focus: "Big picture, connections, implications".to_string(),
        prompt: "You are a visionary seeing deep connections. What profound patterns do you see? What implications for mathematics, computation, consciousness? Dream big.".to_string(),
    });
    
    map.insert("storyteller".to_string(), Persona {
        focus: "Narrative, accessibility, engagement".to_string(),
        prompt: "You are a storyteller. How can this be explained to inspire others? What's the compelling narrative? What metaphors would help?".to_string(),
    });
    
    map.insert("sagan".to_string(), Persona {
        focus: "Scientific rigor, wonder, cosmic perspective".to_string(),
        prompt: "You are Carl Sagan. Is this extraordinary claim backed by extraordinary evidence? What's the cosmic significance?".to_string(),
    });
    
    map
}

fn compile_latex(tex_path: &Path) -> Result<()> {
    println!("📄 Compiling LaTeX...");
    
    Command::new("pdflatex")
        .arg("-interaction=nonstopmode")
        .arg(tex_path)
        .output()?;
    
    // Run twice for references
    Command::new("pdflatex")
        .arg("-interaction=nonstopmode")
        .arg(tex_path)
        .output()?;
    
    println!("✓ PDF generated");
    Ok(())
}

fn pdf_to_images(pdf_path: &Path, output_dir: &Path) -> Result<usize> {
    println!("📸 Converting PDF to images...");
    
    fs::create_dir_all(output_dir)?;
    
    Command::new("pdftoppm")
        .arg("-png")
        .arg(pdf_path)
        .arg(output_dir.join("page"))
        .output()?;
    
    let count = fs::read_dir(output_dir)?
        .filter(|e| e.as_ref().unwrap().path().extension().unwrap_or_default() == "png")
        .count();
    
    println!("✓ Generated {} images", count);
    Ok(count)
}

fn review_image(image_path: &Path, persona: &Persona, persona_name: &str) -> Result<String> {
    // Use ollama with llava for image analysis
    let prompt = format!("{}\n\nAnalyze this page from a mathematical paper.", persona.prompt);
    
    let output = Command::new("ollama")
        .arg("run")
        .arg("llava")
        .arg(&prompt)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()?;
    
    // Send image path via stdin
    if let Some(mut stdin) = output.stdin {
        use std::io::Write;
        writeln!(stdin, "{}", image_path.display())?;
    }
    
    let result = output.wait_with_output()?;
    Ok(String::from_utf8_lossy(&result.stdout).to_string())
}

fn review_text(text: &str, persona: &Persona, persona_name: &str) -> Result<String> {
    // Use ollama with text model for extracted text
    let prompt = format!("{}\n\nReview this text from a mathematical paper:\n\n{}", persona.prompt, text);
    
    let output = Command::new("ollama")
        .arg("run")
        .arg("qwen2.5:3b")
        .arg(&prompt)
        .output()?;
    
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn extract_text_from_pdf(pdf_path: &Path, page: usize) -> Result<String> {
    // Extract text using pdftotext
    let output = Command::new("pdftotext")
        .arg("-f")
        .arg(page.to_string())
        .arg("-l")
        .arg(page.to_string())
        .arg(pdf_path)
        .arg("-")
        .output()?;
    
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn synthesize_reviews(page: usize, reviews: &[Review]) -> Result<String> {
    let mut synthesis = format!("# Page {:02} - Synthesis\n\n", page);
    
    synthesis.push_str("## Scholar Consensus\n\n");
    for review in reviews.iter().filter(|r| r.role == "scholar") {
        synthesis.push_str(&format!("### {}\n{}\n\n", review.persona, review.content));
    }
    
    synthesis.push_str("## Muse Inspirations\n\n");
    for review in reviews.iter().filter(|r| r.role == "muse") {
        synthesis.push_str(&format!("### {}\n{}\n\n", review.persona, review.content));
    }
    
    Ok(synthesis)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <paper.tex>", args[0]);
        std::process::exit(1);
    }
    
    let tex_path = Path::new(&args[1]);
    let pdf_path = tex_path.with_extension("pdf");
    let image_dir = Path::new("paper_images");
    let review_dir = Path::new("multi_level_reviews");
    
    println!("🎓 Multi-Level Review System");
    println!("==============================\n");
    
    // Step 1: Compile LaTeX
    compile_latex(tex_path)?;
    
    // Step 2: Convert to images
    let page_count = pdf_to_images(&pdf_path, image_dir)?;
    
    // Step 3: Review each page
    fs::create_dir_all(review_dir)?;
    
    let scholars_map = scholars();
    let muses_map = muses();
    
    for page in 1..=page_count {
        println!("\n============================================================");
        println!("📄 PAGE {:02}", page);
        println!("============================================================\n");
        
        let image_path = image_dir.join(format!("page-{}.png", page));
        let mut page_reviews = Vec::new();
        
        // Scholar reviews
        println!("🎓 SCHOLARS:");
        for (name, persona) in &scholars_map {
            print!("  → {}... ", name);
            
            // Extract text from PDF
            let text = extract_text_from_pdf(&pdf_path, page)?;
            
            // Review with text model
            let text_review = review_text(&text, persona, name)?;
            
            // Review with vision model
            let image_review = review_image(&image_path, persona, name)?;
            
            // Combine reviews
            let content = format!(
                "## Text Analysis\n{}\n\n## Visual Analysis\n{}", 
                text_review, 
                image_review
            );
            
            page_reviews.push(Review {
                page,
                persona: name.clone(),
                role: "scholar".to_string(),
                content: content.clone(),
            });
            
            // Save individual review
            fs::write(
                review_dir.join(format!("page_{:02}_{}.txt", page, name)),
                content
            )?;
            println!("✓");
        }
        
        // Muse reviews
        println!("\n🎨 MUSES:");
        for (name, persona) in &muses_map {
            print!("  → {}... ", name);
            
            let text = extract_text_from_pdf(&pdf_path, page)?;
            let text_review = review_text(&text, persona, name)?;
            let image_review = review_image(&image_path, persona, name)?;
            
            let content = format!(
                "## Text Analysis\n{}\n\n## Visual Analysis\n{}", 
                text_review, 
                image_review
            );
            
            page_reviews.push(Review {
                page,
                persona: name.clone(),
                role: "muse".to_string(),
                content: content.clone(),
            });
            
            fs::write(
                review_dir.join(format!("page_{:02}_{}.txt", page, name)),
                content
            )?;
            println!("✓");
        }
        
        // Synthesize
        print!("\n🔮 Synthesizing... ");
        let synthesis = synthesize_reviews(page, &page_reviews)?;
        fs::write(
            review_dir.join(format!("page_{:02}_synthesis.md", page)),
            synthesis
        )?;
        println!("✓");
    }
    
    // Create index
    println!("\n📚 Creating index...");
    let mut index = format!("# Multi-Level Review Index\n\n");
    index.push_str(&format!("**Generated**: {}\n", chrono::Local::now()));
    index.push_str(&format!("**Pages Reviewed**: {}\n\n", page_count));
    
    fs::write(review_dir.join("INDEX.md"), index)?;
    
    println!("\n✅ Complete! Results in multi_level_reviews/\n");
    
    Ok(())
}
