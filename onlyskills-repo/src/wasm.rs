// onlyskills.com - zkERDAProlog in WebAssembly (Rust → WASM)

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

#[wasm_bindgen]
#[derive(Debug, Serialize, Deserialize)]
pub struct Skill {
    shard_id: u8,
    prime: u64,
    skill_name: String,
    skill_type: String,
    command: String,
    search_capability: String,
    zkperf_hash: String,
}

const MONSTER_PRIMES: [u64; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];

#[wasm_bindgen]
impl Skill {
    #[wasm_bindgen(constructor)]
    pub fn new(shard_id: u8, name: String, skill_type: String, cmd: String, cap: String, hash: String) -> Self {
        Self {
            shard_id,
            prime: MONSTER_PRIMES[(shard_id % 15) as usize],
            skill_name: name,
            skill_type,
            command: cmd,
            search_capability: cap,
            zkperf_hash: hash,
        }
    }
    
    #[wasm_bindgen(getter)]
    pub fn shard_id(&self) -> u8 { self.shard_id }
    
    #[wasm_bindgen(getter)]
    pub fn prime(&self) -> u64 { self.prime }
    
    #[wasm_bindgen]
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
    
    #[wasm_bindgen]
    pub fn to_rdf(&self) -> String {
        format!(
            "<https://onlyskills.com/skill/{}> rdf:type zkerdfa:SearchSkill .\n\
             <https://onlyskills.com/skill/{}> zkerdfa:shardId {} .\n\
             <https://onlyskills.com/skill/{}> zkerdfa:prime {} .",
            self.skill_name, self.skill_name, self.shard_id, self.skill_name, self.prime
        )
    }
}

#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}

// ∞ 71 Shards in WebAssembly ∞
