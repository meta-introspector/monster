// WASM Archive.org Shard Reader

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};
use serde::{Deserialize, Serialize};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RDFShard {
    pub shard_id: usize,
    pub content_hash: String,
    pub triples: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueLatticeEntry {
    pub value: String,
    pub godel_number: u64,
    pub usage_count: u32,
}

#[wasm_bindgen]
pub struct ArchiveOrgReader {
    item_id: String,
}

#[wasm_bindgen]
impl ArchiveOrgReader {
    #[wasm_bindgen(constructor)]
    pub fn new(item_id: String) -> Self {
        log(&format!("📦 Archive.org reader: {}", item_id));
        Self { item_id }
    }
    
    #[wasm_bindgen]
    pub async fn read_shard(&self, shard_id: usize) -> Result<JsValue, JsValue> {
        let url = format!(
            "https://archive.org/download/{}/monster_shard_{:02}_hash_{}.ttl",
            self.item_id, shard_id, "3083b531"  // First 8 chars of hash
        );
        
        log(&format!("🔍 Fetching: {}", url));
        
        let mut opts = RequestInit::new();
        opts.method("GET");
        opts.mode(RequestMode::Cors);
        
        let request = Request::new_with_str_and_init(&url, &opts)?;
        
        let window = web_sys::window().unwrap();
        let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
        let resp: Response = resp_value.dyn_into()?;
        
        let text = JsFuture::from(resp.text()?).await?;
        let content = text.as_string().unwrap();
        
        let triples: Vec<String> = content.lines()
            .filter(|l| !l.starts_with("@prefix") && !l.is_empty())
            .map(|s| s.to_string())
            .collect();
        
        let shard = RDFShard {
            shard_id,
            content_hash: "computed".to_string(),
            triples,
        };
        
        Ok(serde_wasm_bindgen::to_value(&shard)?)
    }
    
    #[wasm_bindgen]
    pub async fn read_lattice(&self) -> Result<JsValue, JsValue> {
        let url = format!(
            "https://archive.org/download/{}/value_lattice_witnessed.json",
            self.item_id
        );
        
        log(&format!("🔍 Fetching lattice: {}", url));
        
        let mut opts = RequestInit::new();
        opts.method("GET");
        opts.mode(RequestMode::Cors);
        
        let request = Request::new_with_str_and_init(&url, &opts)?;
        
        let window = web_sys::window().unwrap();
        let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
        let resp: Response = resp_value.dyn_into()?;
        
        let json = JsFuture::from(resp.json()?).await?;
        
        Ok(json)
    }
    
    #[wasm_bindgen]
    pub fn get_shard_url(&self, shard_id: usize) -> String {
        format!(
            "https://archive.org/download/{}/monster_shard_{:02}_*.ttl",
            self.item_id, shard_id
        )
    }
}

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}
