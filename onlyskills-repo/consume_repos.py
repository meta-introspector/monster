#!/usr/bin/env python3
"""Consume zos-server, meta-introspector, zombie_driver2 into Monster structure"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib

MONSTER_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]

@dataclass
class ConsumedRepo:
    """Repository consumed into Monster structure"""
    name: str
    path: str
    shard_id: int
    prime: int
    files_count: int
    rust_files: int
    python_files: int
    docs_count: int
    monster_form: str
    hecke_signature: int

def consume_repo(name: str, path: str) -> ConsumedRepo:
    """Consume a repository into Monster structure"""
    repo_path = Path(path).expanduser()
    
    if not repo_path.exists():
        print(f"⚠️  {name} not found at {path}")
        return None
    
    # Count files
    files = list(repo_path.rglob("*"))
    files_count = len([f for f in files if f.is_file()])
    rust_files = len(list(repo_path.rglob("*.rs")))
    python_files = len(list(repo_path.rglob("*.py")))
    docs_count = len(list(repo_path.rglob("*.md")))
    
    # Hash to shard
    repo_hash = hashlib.sha256(name.encode()).hexdigest()
    shard_id = int(repo_hash[:8], 16) % 71
    prime = MONSTER_PRIMES[shard_id % 15]
    
    # Determine Monster form
    if rust_files > python_files:
        monster_form = "Rust Manifold"
    elif python_files > rust_files:
        monster_form = "Python Fiber Bundle"
    else:
        monster_form = "Polyglot Module"
    
    # Compute Hecke signature
    signature = (
        files_count * MONSTER_PRIMES[0] +
        rust_files * MONSTER_PRIMES[1] +
        python_files * MONSTER_PRIMES[2] +
        docs_count * MONSTER_PRIMES[3]
    )
    
    return ConsumedRepo(
        name=name,
        path=str(repo_path),
        shard_id=shard_id,
        prime=prime,
        files_count=files_count,
        rust_files=rust_files,
        python_files=python_files,
        docs_count=docs_count,
        monster_form=monster_form,
        hecke_signature=signature
    )

def main():
    print("🌊 Consuming Repositories into Monster Structure")
    print("=" * 70)
    
    repos = [
        ("zos-server", "~/terraform/services/submodules/zos-server"),
        ("meta-introspector", "/mnt/data1/meta-introspector"),
        ("zombie_driver2", "~/nix/vendor/rust/cargo2nix/submodules/rust-build/compiler/zombie_driver2"),
    ]
    
    consumed = []
    
    for name, path in repos:
        print(f"\n📦 Consuming {name}...")
        repo = consume_repo(name, path)
        if repo:
            consumed.append(repo)
            print(f"   Shard: {repo.shard_id}")
            print(f"   Prime: {repo.prime}")
            print(f"   Files: {repo.files_count}")
            print(f"   Rust: {repo.rust_files}")
            print(f"   Python: {repo.python_files}")
            print(f"   Docs: {repo.docs_count}")
            print(f"   Form: {repo.monster_form}")
            print(f"   Signature: {repo.hecke_signature}")
    
    # Save
    output = {
        "consumed_repos": [asdict(r) for r in consumed],
        "total_files": sum(r.files_count for r in consumed),
        "total_rust": sum(r.rust_files for r in consumed),
        "total_python": sum(r.python_files for r in consumed),
        "total_docs": sum(r.docs_count for r in consumed),
    }
    
    Path("consumed_repos.json").write_text(json.dumps(output, indent=2))
    
    print("\n" + "=" * 70)
    print("📊 Summary:")
    print(f"  Repos consumed: {len(consumed)}")
    print(f"  Total files: {output['total_files']:,}")
    print(f"  Total Rust: {output['total_rust']:,}")
    print(f"  Total Python: {output['total_python']:,}")
    print(f"  Total Docs: {output['total_docs']:,}")
    
    print(f"\n💾 Saved to consumed_repos.json")
    print("\n∞ Repos Consumed. Monster Structure Complete. ∞")

if __name__ == "__main__":
    main()
