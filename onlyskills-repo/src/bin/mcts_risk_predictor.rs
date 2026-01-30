// zkprologml: ML-based MCTS Risk Prediction Market
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const MONSTER_PRIMES: [u32; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MCTSNode {
    state: String,
    visits: u32,
    value: f64,
    children: Vec<usize>,
    parent: Option<usize>,
    risk_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PredictionMarket {
    asset: String,
    chain: String,
    repo_url: String,
    risk_probability: f64,
    confidence: f64,
    shard_id: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RiskAssessment {
    threat_level: String,
    zone: u8,
    mcts_score: f64,
    market_price: f64,
}

impl MCTSNode {
    fn new(state: String) -> Self {
        Self {
            state,
            visits: 0,
            value: 0.0,
            children: Vec::new(),
            parent: None,
            risk_score: 0.0,
        }
    }
    
    fn ucb1(&self, parent_visits: u32, exploration: f64) -> f64 {
        if self.visits == 0 {
            return f64::INFINITY;
        }
        
        let exploitation = self.value / self.visits as f64;
        let exploration_term = exploration * ((parent_visits as f64).ln() / self.visits as f64).sqrt();
        
        exploitation + exploration_term
    }
}

struct MCTSRiskPredictor {
    nodes: Vec<MCTSNode>,
    root: usize,
    exploration_constant: f64,
}

impl MCTSRiskPredictor {
    fn new(initial_state: String) -> Self {
        let root_node = MCTSNode::new(initial_state);
        Self {
            nodes: vec![root_node],
            root: 0,
            exploration_constant: 1.414, // sqrt(2)
        }
    }
    
    fn select(&self, node_idx: usize) -> usize {
        let node = &self.nodes[node_idx];
        
        if node.children.is_empty() {
            return node_idx;
        }
        
        let parent_visits = node.visits;
        let best_child = node.children.iter()
            .max_by(|&&a, &&b| {
                let score_a = self.nodes[a].ucb1(parent_visits, self.exploration_constant);
                let score_b = self.nodes[b].ucb1(parent_visits, self.exploration_constant);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .copied()
            .unwrap();
        
        self.select(best_child)
    }
    
    fn expand(&mut self, node_idx: usize, new_states: Vec<String>) {
        for state in new_states {
            let mut child = MCTSNode::new(state);
            child.parent = Some(node_idx);
            
            let child_idx = self.nodes.len();
            self.nodes.push(child);
            self.nodes[node_idx].children.push(child_idx);
        }
    }
    
    fn simulate(&self, node_idx: usize) -> f64 {
        // Simulate risk outcome
        let node = &self.nodes[node_idx];
        
        // Risk factors
        let mut risk = 0.0;
        
        if node.state.contains("unsafe") { risk += 0.3; }
        if node.state.contains("network") { risk += 0.2; }
        if node.state.contains("spawn") { risk += 0.2; }
        if node.state.contains("few_commits") { risk += 0.2; }
        if node.state.contains("few_authors") { risk += 0.1; }
        
        risk.min(1.0)
    }
    
    fn backpropagate(&mut self, mut node_idx: usize, value: f64) {
        loop {
            let node = &mut self.nodes[node_idx];
            node.visits += 1;
            node.value += value;
            node.risk_score = node.value / node.visits as f64;
            
            match node.parent {
                Some(parent_idx) => node_idx = parent_idx,
                None => break,
            }
        }
    }
    
    fn run_iterations(&mut self, iterations: u32) {
        for _ in 0..iterations {
            let selected = self.select(self.root);
            
            // Expand if not terminal
            if self.nodes[selected].children.is_empty() {
                let new_states = vec![
                    format!("{}_safe", self.nodes[selected].state),
                    format!("{}_risky", self.nodes[selected].state),
                ];
                self.expand(selected, new_states);
            }
            
            let value = self.simulate(selected);
            self.backpropagate(selected, value);
        }
    }
    
    fn best_action(&self) -> (usize, f64) {
        let root = &self.nodes[self.root];
        
        let best = root.children.iter()
            .max_by(|&&a, &&b| {
                self.nodes[a].visits.cmp(&self.nodes[b].visits)
            })
            .copied()
            .unwrap_or(self.root);
        
        (best, self.nodes[best].risk_score)
    }
}

fn assess_risk_mcts(asset: &str, chain: &str, repo_url: &str) -> RiskAssessment {
    let initial_state = format!("{}_{}", chain, asset);
    let mut predictor = MCTSRiskPredictor::new(initial_state);
    
    // Run MCTS
    predictor.run_iterations(1000);
    
    let (_, risk_score) = predictor.best_action();
    
    // Map to threat level and zone
    let (threat_level, zone) = if risk_score > 0.8 {
        ("catastrophic", 71)
    } else if risk_score > 0.6 {
        ("critical", 59)
    } else if risk_score > 0.4 {
        ("high", 47)
    } else if risk_score > 0.2 {
        ("medium", 31)
    } else {
        ("low", 11)
    };
    
    // Market price = risk probability
    let market_price = risk_score;
    
    RiskAssessment {
        threat_level: threat_level.to_string(),
        zone,
        mcts_score: risk_score,
        market_price,
    }
}

fn create_prediction_market(asset: &str, chain: &str, repo_url: &str) -> PredictionMarket {
    let risk = assess_risk_mcts(asset, chain, repo_url);
    
    // Assign to shard based on zone
    let shard_id = MONSTER_PRIMES.iter()
        .position(|&p| p == risk.zone as u32)
        .unwrap_or(0) as u8;
    
    PredictionMarket {
        asset: asset.to_string(),
        chain: chain.to_string(),
        repo_url: repo_url.to_string(),
        risk_probability: risk.mcts_score,
        confidence: 1.0 - (risk.mcts_score - 0.5).abs() * 2.0,
        shard_id,
    }
}

fn main() {
    println!("🎲 zkprologml: MCTS Risk Prediction Market");
    println!("{}", "=".repeat(70));
    println!();
    
    // Analyze top coins
    let coins = vec![
        ("SOL", "solana", "https://github.com/solana-labs/solana"),
        ("BONK", "solana", "https://github.com/bonk-inu/bonk"),
        ("ETH", "ethereum", "https://github.com/ethereum/go-ethereum"),
        ("UNI", "ethereum", "https://github.com/Uniswap/v3-core"),
    ];
    
    println!("📊 Creating prediction markets...");
    let mut markets = Vec::new();
    
    for (asset, chain, repo) in coins {
        let market = create_prediction_market(asset, chain, repo);
        println!("  {} ({}) - Risk: {:.2}% - Shard: {}", 
                 asset, chain, market.risk_probability * 100.0, market.shard_id);
        markets.push(market);
    }
    
    println!();
    
    // Find the lobster (lowest risk, highest confidence)
    println!("🦞 Finding the lobster...");
    let lobster = markets.iter()
        .min_by(|a, b| a.risk_probability.partial_cmp(&b.risk_probability).unwrap())
        .unwrap();
    
    println!("  Asset: {}", lobster.asset);
    println!("  Chain: {}", lobster.chain);
    println!("  Risk: {:.2}%", lobster.risk_probability * 100.0);
    println!("  Confidence: {:.2}%", lobster.confidence * 100.0);
    println!("  Shard: {}", lobster.shard_id);
    println!();
    
    // Save markets
    let json = serde_json::to_string_pretty(&markets).unwrap();
    std::fs::write("mcts_prediction_markets.json", json).unwrap();
    
    println!("💾 Saved: mcts_prediction_markets.json");
    println!();
    println!("∞ MCTS Risk Predicted. Markets Created. Lobster Found. ∞");
}
