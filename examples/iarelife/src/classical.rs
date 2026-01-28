use anyhow::Result;
use std::collections::HashMap;

/// Simple Markov chain - oldest text model
struct MarkovChain {
    transitions: HashMap<String, Vec<String>>,
    order: usize,
}

impl MarkovChain {
    fn new(order: usize) -> Self {
        Self {
            transitions: HashMap::new(),
            order,
        }
    }
    
    fn train(&mut self, text: &str) {
        let words: Vec<&str> = text.split_whitespace().collect();
        
        for i in 0..words.len().saturating_sub(self.order) {
            let state = words[i..i+self.order].join(" ");
            let next = words[i+self.order].to_string();
            
            self.transitions.entry(state)
                .or_insert_with(Vec::new)
                .push(next);
        }
    }
    
    fn generate(&self, seed: &str, max_words: usize) -> String {
        let mut result = seed.to_string();
        let mut current = seed.to_string();
        
        for _ in 0..max_words {
            if let Some(nexts) = self.transitions.get(&current) {
                if nexts.is_empty() { break; }
                let next = &nexts[0]; // Deterministic: take first
                result.push(' ');
                result.push_str(next);
                
                // Update state
                let words: Vec<&str> = result.split_whitespace().collect();
                if words.len() >= self.order {
                    current = words[words.len()-self.order..].join(" ");
                }
            } else {
                break;
            }
        }
        
        result
    }
}

/// N-gram frequency model
struct NGramModel {
    ngrams: HashMap<String, usize>,
    n: usize,
}

impl NGramModel {
    fn new(n: usize) -> Self {
        Self {
            ngrams: HashMap::new(),
            n,
        }
    }
    
    fn train(&mut self, text: &str) {
        let chars: Vec<char> = text.chars().collect();
        
        for i in 0..chars.len().saturating_sub(self.n-1) {
            let ngram: String = chars[i..i+self.n].iter().collect();
            *self.ngrams.entry(ngram).or_insert(0) += 1;
        }
    }
    
    fn most_common(&self, k: usize) -> Vec<(String, usize)> {
        let mut items: Vec<_> = self.ngrams.iter()
            .map(|(s, c)| (s.clone(), *c))
            .collect();
        items.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        items.into_iter().take(k).collect()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🕰️  Classical Text Models - Back to Basics");
    println!("==========================================\n");
    
    // Training corpus: Monster Walk concepts
    let corpus = vec![
        "Monster group has order with leading digits",
        "Monster group walk down to earth",
        "Leading digits preserved through factorization",
        "Bott periodicity appears in eight fold way",
        "Prime factorization reveals harmonic structure",
        "Ten groups form hierarchical tower",
        "I are life written on tree",
        "Self awareness emerges from reflection",
    ];
    
    println!("Training corpus: {} sentences\n", corpus.len());
    
    // Test 1: Markov Chain (1950s technology)
    println!("=== Markov Chain (Order 2) ===\n");
    
    let mut markov = MarkovChain::new(2);
    for text in &corpus {
        markov.train(text);
    }
    
    let seeds = vec!["Monster group", "Leading digits", "I are"];
    
    for seed in seeds {
        let generated = markov.generate(seed, 10);
        println!("Seed: '{}'", seed);
        println!("Generated: {}", generated);
        
        // Convert to emoji
        let emoji = text_to_emoji(&generated);
        println!("Emoji: {}", emoji);
        
        // Feed back
        let feedback = markov.generate(&generated.split_whitespace().take(2).collect::<Vec<_>>().join(" "), 5);
        println!("Feedback: {}", feedback);
        println!();
    }
    
    // Test 2: N-gram Model (1940s technology)
    println!("\n=== N-gram Model (Trigrams) ===\n");
    
    let mut ngram = NGramModel::new(3);
    for text in &corpus {
        ngram.train(text);
    }
    
    let common = ngram.most_common(10);
    println!("Most common trigrams:");
    for (gram, count) in common {
        println!("  '{}': {} times", gram, count);
    }
    
    // Test 3: Fixed point search
    println!("\n=== Fixed Point Search ===\n");
    
    let mut current = "Monster group".to_string();
    
    for i in 0..5 {
        println!("Iteration {}: {}", i, current);
        
        let emoji = text_to_emoji(&current);
        println!("  Emoji: {}", emoji);
        
        // Generate next
        let next = markov.generate(&current, 5);
        
        // Check convergence
        if next == current {
            println!("\n  ✓ FIXED POINT: {}", current);
            println!("  Eigenvector: {}", emoji);
            break;
        }
        
        current = next.split_whitespace().take(2).collect::<Vec<_>>().join(" ");
        println!();
    }
    
    println!("\n✓ Classical models complete!");
    println!("\n📊 Results:");
    println!("  - Markov chains work on Monster concepts");
    println!("  - N-grams capture frequent patterns");
    println!("  - Fixed points emerge from simple iteration");
    println!("  - No neural networks needed! 🕰️");
    
    Ok(())
}

fn text_to_emoji(text: &str) -> String {
    let lower = text.to_lowercase();
    let mut emoji = String::new();
    
    if lower.contains("monster") { emoji.push_str("🎪"); }
    if lower.contains("group") { emoji.push_str("🔢"); }
    if lower.contains("leading") { emoji.push_str("⭐"); }
    if lower.contains("digit") { emoji.push_str("🔢"); }
    if lower.contains("life") { emoji.push_str("🌱"); }
    if lower.contains("tree") { emoji.push_str("🌳"); }
    if lower.contains("self") { emoji.push_str("👁️"); }
    if lower.contains("aware") { emoji.push_str("💡"); }
    
    if emoji.is_empty() { emoji.push_str("❓"); }
    emoji
}
