// onlyskills.com - zkERDAProlog in Rust

pub mod gpu_monster;

use serde::{Deserialize, Serialize};

const MONSTER_PRIMES: [u64; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];

#[derive(Debug, Serialize, Deserialize)]
pub struct Skill {
    pub shard_id: u8,
    pub prime: u64,
    pub skill_name: String,
    pub skill_type: String,
    pub command: String,
    pub search_capability: String,
    pub zkperf_hash: String,
}

impl Skill {
    pub fn new(shard_id: u8, name: &str, skill_type: &str, cmd: &str, cap: &str, hash: &str) -> Self {
        Self {
            shard_id,
            prime: MONSTER_PRIMES[(shard_id % 15) as usize],
            skill_name: name.to_string(),
            skill_type: skill_type.to_string(),
            command: cmd.to_string(),
            search_capability: cap.to_string(),
            zkperf_hash: hash.to_string(),
        }
    }
    
    pub fn to_rdf(&self) -> String {
        format!(
            "<https://onlyskills.com/skill/{}> rdf:type zkerdfa:SearchSkill .\n\
             <https://onlyskills.com/skill/{}> zkerdfa:shardId {} .\n\
             <https://onlyskills.com/skill/{}> zkerdfa:prime {} .",
            self.skill_name, self.skill_name, self.shard_id, self.skill_name, self.prime
        )
    }
}

fn main() {
    let skill = Skill::new(29, "expert_system", "search_explicit_search", 
                           "cargo run --release --bin expert_system", 
                           "explicit_search", "a3f5b2c1d4e6f7a8");
    
    println!("🦀 Rust zkERDAProlog Skill Registry");
    println!("JSON: {}", serde_json::to_string(&skill).unwrap());
    println!("RDF:\n{}", skill.to_rdf());
    println!("∞ 71 Shards in Rust ∞");
}
