// Import OpenClaw into onlyskills and shard into Monster
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const MONSTER_PRIMES: [u8; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];

#[derive(Debug, Clone, Serialize)]
struct Fork {
    id: u32,
    name: String,
    owner: String,
    shard: u8,
}

#[derive(Debug, Clone, Serialize)]
struct PullRequest {
    number: u32,
    title: String,
    author: String,
    shard: u8,
}

#[derive(Debug, Clone, Serialize)]
struct Issue {
    number: u32,
    title: String,
    labels: Vec<String>,
    shard: u8,
}

#[derive(Debug, Clone, Serialize)]
struct CodeFile {
    path: String,
    size: usize,
    shard: u8,
}

#[derive(Debug, Clone, Serialize)]
struct Author {
    name: String,
    commits: u32,
    shard: u8,
}

#[derive(Debug, Clone, Serialize)]
struct OpenClawImport {
    repo: String,
    forks: Vec<Fork>,
    prs: Vec<PullRequest>,
    issues: Vec<Issue>,
    code: Vec<CodeFile>,
    authors: Vec<Author>,
    shard_distribution: HashMap<u8, u32>,
}

fn hash_to_shard(data: &str) -> u8 {
    let sum: u32 = data.bytes().map(|b| b as u32).sum();
    (sum % 71) as u8
}

fn commits_to_shard(commits: u32) -> u8 {
    if commits < 10 { 2 }
    else if commits < 50 { 11 }
    else if commits < 100 { 23 }
    else if commits < 500 { 47 }
    else { 71 }
}

fn shard_forks(forks: Vec<(u32, String, String)>) -> Vec<Fork> {
    forks.into_iter().map(|(id, name, owner)| {
        let shard = hash_to_shard(&format!("{}{}", name, owner));
        Fork { id, name, owner, shard }
    }).collect()
}

fn shard_prs(prs: Vec<(u32, String, String)>) -> Vec<PullRequest> {
    prs.into_iter().map(|(number, title, author)| {
        let shard = hash_to_shard(&author);
        PullRequest { number, title, author, shard }
    }).collect()
}

fn shard_issues(issues: Vec<(u32, String, Vec<String>)>) -> Vec<Issue> {
    issues.into_iter().map(|(number, title, labels)| {
        let label_str = labels.join(",");
        let shard = hash_to_shard(&label_str);
        Issue { number, title, labels, shard }
    }).collect()
}

fn shard_code(files: Vec<(String, usize)>) -> Vec<CodeFile> {
    files.into_iter().map(|(path, size)| {
        let shard = hash_to_shard(&path);
        CodeFile { path, size, shard }
    }).collect()
}

fn shard_authors(authors: Vec<(String, u32)>) -> Vec<Author> {
    authors.into_iter().map(|(name, commits)| {
        let shard = commits_to_shard(commits);
        Author { name, commits, shard }
    }).collect()
}

fn calculate_distribution(import: &OpenClawImport) -> HashMap<u8, u32> {
    let mut dist = HashMap::new();
    
    for fork in &import.forks {
        *dist.entry(fork.shard).or_insert(0) += 1;
    }
    for pr in &import.prs {
        *dist.entry(pr.shard).or_insert(0) += 1;
    }
    for issue in &import.issues {
        *dist.entry(issue.shard).or_insert(0) += 1;
    }
    for file in &import.code {
        *dist.entry(file.shard).or_insert(0) += 1;
    }
    for author in &import.authors {
        *dist.entry(author.shard).or_insert(0) += 1;
    }
    
    dist
}

fn import_openclaw() -> OpenClawImport {
    let repo = "https://github.com/steipete/openclaw".to_string();
    
    // Simulate GitHub API data
    let fork_data = vec![
        (1, "openclaw-fork-1".to_string(), "user1".to_string()),
        (2, "openclaw-fork-2".to_string(), "user2".to_string()),
        (3, "openclaw-fork-3".to_string(), "user3".to_string()),
    ];
    
    let pr_data = vec![
        (1, "Add feature X".to_string(), "steipete".to_string()),
        (2, "Fix bug Y".to_string(), "contributor1".to_string()),
        (3, "Update docs".to_string(), "contributor2".to_string()),
    ];
    
    let issue_data = vec![
        (1, "Bug report".to_string(), vec!["bug".to_string()]),
        (2, "Feature request".to_string(), vec!["enhancement".to_string()]),
        (3, "Question".to_string(), vec!["question".to_string()]),
    ];
    
    let code_data = vec![
        ("src/main.rs".to_string(), 1024),
        ("src/lib.rs".to_string(), 2048),
        ("README.md".to_string(), 512),
    ];
    
    let author_data = vec![
        ("steipete".to_string(), 500),
        ("contributor1".to_string(), 50),
        ("contributor2".to_string(), 10),
    ];
    
    let forks = shard_forks(fork_data);
    let prs = shard_prs(pr_data);
    let issues = shard_issues(issue_data);
    let code = shard_code(code_data);
    let authors = shard_authors(author_data);
    
    let mut import = OpenClawImport {
        repo,
        forks,
        prs,
        issues,
        code,
        authors,
        shard_distribution: HashMap::new(),
    };
    
    import.shard_distribution = calculate_distribution(&import);
    
    import
}

fn main() {
    println!("📦 Importing OpenClaw into onlyskills");
    println!("{}", "=".repeat(70));
    println!();
    
    let import = import_openclaw();
    
    println!("📊 Import Summary:");
    println!("  Repo: {}", import.repo);
    println!("  Forks: {}", import.forks.len());
    println!("  PRs: {}", import.prs.len());
    println!("  Issues: {}", import.issues.len());
    println!("  Code files: {}", import.code.len());
    println!("  Authors: {}", import.authors.len());
    println!();
    
    println!("🔀 Forks:");
    for fork in &import.forks {
        println!("  {} by {} → Shard {}", fork.name, fork.owner, fork.shard);
    }
    println!();
    
    println!("🔧 Pull Requests:");
    for pr in &import.prs {
        println!("  #{} by {} → Shard {}", pr.number, pr.author, pr.shard);
    }
    println!();
    
    println!("🐛 Issues:");
    for issue in &import.issues {
        println!("  #{} ({:?}) → Shard {}", issue.number, issue.labels, issue.shard);
    }
    println!();
    
    println!("📄 Code:");
    for file in &import.code {
        println!("  {} ({} bytes) → Shard {}", file.path, file.size, file.shard);
    }
    println!();
    
    println!("👥 Authors:");
    for author in &import.authors {
        println!("  {} ({} commits) → Shard {}", author.name, author.commits, author.shard);
    }
    println!();
    
    println!("📊 Shard Distribution:");
    let mut shards: Vec<_> = import.shard_distribution.iter().collect();
    shards.sort_by_key(|(shard, _)| *shard);
    for (shard, count) in shards {
        println!("  Shard {}: {} items", shard, count);
    }
    println!();
    
    // Save import
    let json = serde_json::to_string_pretty(&import).unwrap();
    std::fs::write("openclaw_import.json", json).unwrap();
    
    println!("💾 Saved: openclaw_import.json");
    println!();
    println!("∞ OpenClaw Imported. All Components Sharded. Monster Ready. ∞");
}
