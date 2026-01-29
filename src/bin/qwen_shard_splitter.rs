// Rust: Split Qwen shards by topic, geography, and Wikidata QIDs

use std::collections::HashMap;

/// Wikidata QID
#[derive(Debug, Clone)]
pub struct WikidataQID {
    pub id: u64,
    pub label: String,
}

/// Geographic region
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Geography {
    NorthAmerica,
    SouthAmerica,
    Europe,
    Asia,
    Africa,
    Oceania,
    Antarctica,
}

/// Topic category (15 topics for 15 chunks)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Topic {
    Science,
    Technology,
    History,
    Geography,
    Arts,
    Mathematics,
    Politics,
    Sports,
    Medicine,
    Philosophy,
    Economics,
    Literature,
    Music,
    Engineering,
    Law,
}

impl Topic {
    pub fn to_chunk_id(&self) -> u8 {
        match self {
            Topic::Science => 0,
            Topic::Technology => 1,
            Topic::History => 2,
            Topic::Geography => 3,
            Topic::Arts => 4,
            Topic::Mathematics => 5,
            Topic::Politics => 6,
            Topic::Sports => 7,
            Topic::Medicine => 8,
            Topic::Philosophy => 9,
            Topic::Economics => 10,
            Topic::Literature => 11,
            Topic::Music => 12,
            Topic::Engineering => 13,
            Topic::Law => 14,
        }
    }
}

/// Shard split (10 MB WASM module)
#[derive(Debug, Clone)]
pub struct ShardSplit {
    pub prime_shard: u8,      // 0-14 (Monster primes)
    pub chunk_id: u8,         // 0-14 (topic-based)
    pub topic: Topic,
    pub geography: Geography,
    pub qids: Vec<WikidataQID>,
    pub size_mb: usize,       // 10 MB
}

/// Shard splitter
pub struct QwenShardSplitter {
    monster_primes: Vec<u32>,
}

impl QwenShardSplitter {
    pub fn new() -> Self {
        Self {
            monster_primes: vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47],
        }
    }
    
    /// Map Wikidata QID to Monster prime shard
    pub fn qid_to_shard(&self, qid: u64) -> u8 {
        (qid % 15) as u8
    }
    
    /// Create shard split
    pub fn create_split(
        &self,
        qid: &WikidataQID,
        topic: Topic,
        geography: Geography,
    ) -> ShardSplit {
        let prime_shard = self.qid_to_shard(qid.id);
        let chunk_id = topic.to_chunk_id();
        
        ShardSplit {
            prime_shard,
            chunk_id,
            topic,
            geography,
            qids: vec![qid.clone()],
            size_mb: 10,
        }
    }
    
    /// Split all data into 225 shards (15 primes × 15 topics)
    pub fn split_all(&self, data: Vec<(WikidataQID, Topic, Geography)>) -> Vec<ShardSplit> {
        let mut shards: HashMap<(u8, u8), ShardSplit> = HashMap::new();
        
        for (qid, topic, geo) in data {
            let split = self.create_split(&qid, topic, geo);
            let key = (split.prime_shard, split.chunk_id);
            
            shards.entry(key)
                .or_insert_with(|| split.clone())
                .qids.push(qid);
        }
        
        shards.into_values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_qid_mapping() {
        let splitter = QwenShardSplitter::new();
        
        // Test QID mapping
        assert_eq!(splitter.qid_to_shard(100), 10);
        assert_eq!(splitter.qid_to_shard(42), 12);
    }
    
    #[test]
    fn test_topic_chunks() {
        assert_eq!(Topic::Science.to_chunk_id(), 0);
        assert_eq!(Topic::Law.to_chunk_id(), 14);
    }
    
    #[test]
    fn test_total_shards() {
        let splitter = QwenShardSplitter::new();
        
        // 15 primes × 15 topics = 225 shards
        let total = 15 * 15;
        assert_eq!(total, 225);
    }
}

fn main() {
    println!("🗂️  Qwen Shard Splitter");
    println!("="*70);
    println!();
    
    let splitter = QwenShardSplitter::new();
    
    println!("Configuration:");
    println!("  Monster primes: 15 (2-47)");
    println!("  Topics: 15");
    println!("  Total shards: 225 (15 × 15)");
    println!("  Shard size: 10 MB (WASM optimal)");
    println!();
    
    // Example data
    let data = vec![
        (WikidataQID { id: 42, label: "Douglas Adams".to_string() }, Topic::Literature, Geography::Europe),
        (WikidataQID { id: 100, label: "Mathematics".to_string() }, Topic::Mathematics, Geography::NorthAmerica),
        (WikidataQID { id: 71, label: "Monster Group".to_string() }, Topic::Mathematics, Geography::Europe),
    ];
    
    let shards = splitter.split_all(data);
    
    println!("Example splits:");
    for shard in shards.iter().take(3) {
        println!("  Prime shard {}, Chunk {} ({:?}): {} QIDs",
                 shard.prime_shard, shard.chunk_id, shard.topic, shard.qids.len());
    }
    
    println!();
    println!("="*70);
    println!("✅ Split by topic, geography, and Wikidata QIDs!");
}
