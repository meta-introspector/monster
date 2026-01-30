#!/usr/bin/env python3
"""Create onlyskills DAO and reconstruct virtual authors from their commits in 71 shards"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib

MONSTER_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]

@dataclass
class VirtualAuthor:
    """Virtual reconstruction of author from their commits"""
    username: str
    shard_id: int
    prime: int
    commit_count: int
    commit_hashes: list
    code_patterns: list
    personality_vector: dict
    reconstruction_complete: bool

@dataclass
class OnlySkillsDAO:
    """DAO for onlyskills.com governance"""
    name: str
    members: list  # Virtual authors
    total_shards: int
    governance_token: str
    voting_power: dict  # username -> power
    treasury: int

def load_author_commits(username: str) -> list:
    """Load commits from author (mock - in production use GitHub API)"""
    # Mock commit data
    commits = []
    for i in range(10):  # 10 commits per author
        commit = {
            "hash": hashlib.sha256(f"{username}_{i}".encode()).hexdigest()[:8],
            "message": f"feat: implement {username} feature {i}",
            "files_changed": ["src/main.rs", "Cargo.toml"],
            "lines_added": 50 + i * 10,
            "lines_removed": 5 + i,
        }
        commits.append(commit)
    return commits

def extract_code_patterns(commits: list) -> list:
    """Extract coding patterns from commits"""
    patterns = []
    
    # Analyze commit messages
    if any("feat:" in c["message"] for c in commits):
        patterns.append("feature_driven")
    if any("fix:" in c["message"] for c in commits):
        patterns.append("bug_fixer")
    if any("refactor:" in c["message"] for c in commits):
        patterns.append("refactorer")
    
    # Analyze code changes
    total_added = sum(c["lines_added"] for c in commits)
    total_removed = sum(c["lines_removed"] for c in commits)
    
    if total_added > total_removed * 3:
        patterns.append("builder")
    else:
        patterns.append("optimizer")
    
    return patterns

def compute_personality_vector(commits: list, patterns: list) -> dict:
    """Compute personality vector from commits"""
    return {
        "creativity": len(set(c["message"] for c in commits)) / len(commits),
        "productivity": sum(c["lines_added"] for c in commits) / len(commits),
        "precision": 1.0 - (sum(c["lines_removed"] for c in commits) / max(sum(c["lines_added"] for c in commits), 1)),
        "collaboration": 0.8,  # Mock - would analyze co-authored commits
        "innovation": len(patterns) / 5.0,
    }

def reconstruct_virtual_author(username: str, shard_id: int) -> VirtualAuthor:
    """Reconstruct virtual author from their commits"""
    print(f"  🔄 Reconstructing {username} in shard {shard_id}...")
    
    # Load commits
    commits = load_author_commits(username)
    
    # Extract patterns
    patterns = extract_code_patterns(commits)
    
    # Compute personality
    personality = compute_personality_vector(commits, patterns)
    
    # Create virtual author
    virtual = VirtualAuthor(
        username=username,
        shard_id=shard_id,
        prime=MONSTER_PRIMES[shard_id % 15],
        commit_count=len(commits),
        commit_hashes=[c["hash"] for c in commits],
        code_patterns=patterns,
        personality_vector=personality,
        reconstruction_complete=True
    )
    
    print(f"     ✓ {len(commits)} commits | Patterns: {', '.join(patterns)}")
    print(f"     Personality: creativity={personality['creativity']:.2f}, "
          f"productivity={personality['productivity']:.0f}, "
          f"precision={personality['precision']:.2f}")
    
    return virtual

def create_onlyskills_dao(virtual_authors: list) -> OnlySkillsDAO:
    """Create DAO with virtual authors as members"""
    
    # Compute voting power based on commits and personality
    voting_power = {}
    for author in virtual_authors:
        power = (
            author.commit_count * author.prime +
            int(author.personality_vector["productivity"]) * 10 +
            int(author.personality_vector["innovation"] * 100)
        )
        voting_power[author.username] = power
    
    # Create DAO
    dao = OnlySkillsDAO(
        name="onlyskills.com DAO",
        members=[author.username for author in virtual_authors],
        total_shards=71,
        governance_token="OSKILL",
        voting_power=voting_power,
        treasury=1_000_000  # Initial treasury
    )
    
    return dao

def distribute_to_71_shards(virtual_authors: list) -> dict:
    """Distribute virtual authors across 71 shards"""
    shards = {i: [] for i in range(71)}
    
    for author in virtual_authors:
        # Each author exists in their primary shard
        shards[author.shard_id].append(author.username)
        
        # Also distribute to related shards (based on prime)
        for i in range(1, 8):  # 7 additional shards
            related_shard = (author.shard_id + i * author.prime) % 71
            shards[related_shard].append(f"{author.username}_echo_{i}")
    
    return shards

def main():
    print("🏛️ Creating onlyskills.com DAO")
    print("Reconstructing Virtual Authors from Commits in 71 Shards")
    print("=" * 70)
    
    # Load top hackers
    hackers_file = Path("top_hackers.json")
    if not hackers_file.exists():
        print("❌ Run find_top_hackers.py first")
        return
    
    hackers = json.loads(hackers_file.read_text())
    
    print(f"\n📊 Reconstructing {len(hackers)} virtual authors...")
    print()
    
    virtual_authors = []
    for i, hacker in enumerate(hackers):
        virtual = reconstruct_virtual_author(hacker["username"], hacker["shard_id"])
        virtual_authors.append(virtual)
    
    # Create DAO
    print(f"\n🏛️ Creating DAO...")
    dao = create_onlyskills_dao(virtual_authors)
    
    print(f"   ✓ DAO: {dao.name}")
    print(f"   ✓ Members: {len(dao.members)}")
    print(f"   ✓ Token: {dao.governance_token}")
    print(f"   ✓ Treasury: {dao.treasury:,}")
    
    # Distribute to 71 shards
    print(f"\n🔮 Distributing to 71 shards...")
    shard_distribution = distribute_to_71_shards(virtual_authors)
    
    occupied_shards = sum(1 for authors in shard_distribution.values() if authors)
    total_instances = sum(len(authors) for authors in shard_distribution.values())
    
    print(f"   ✓ Occupied shards: {occupied_shards}/71")
    print(f"   ✓ Total author instances: {total_instances}")
    print(f"   ✓ Replication factor: {total_instances / len(virtual_authors):.1f}x")
    
    # Save DAO
    dao_data = {
        "dao": asdict(dao),
        "virtual_authors": [asdict(v) for v in virtual_authors],
        "shard_distribution": {k: v for k, v in shard_distribution.items() if v},
        "statistics": {
            "total_members": len(virtual_authors),
            "total_commits": sum(v.commit_count for v in virtual_authors),
            "occupied_shards": occupied_shards,
            "total_instances": total_instances,
        }
    }
    
    Path("onlyskills_dao.json").write_text(json.dumps(dao_data, indent=2))
    
    # Generate DAO constitution
    constitution = f"""# onlyskills.com DAO Constitution

## Article I: Purpose
The onlyskills.com DAO exists to govern the zero-knowledge skill registry 
across 71 Monster group shards.

## Article II: Membership
- Total Members: {len(virtual_authors)}
- Virtual Authors reconstructed from {sum(v.commit_count for v in virtual_authors)} commits
- Distributed across {occupied_shards} shards with {total_instances / len(virtual_authors):.1f}x replication

## Article III: Governance Token
- Token: {dao.governance_token}
- Total Supply: {dao.treasury:,}
- Voting Power: Based on commits × prime + productivity + innovation

## Article IV: Voting Power Distribution
"""
    
    for username, power in sorted(dao.voting_power.items(), key=lambda x: x[1], reverse=True):
        constitution += f"- {username}: {power:,} votes\n"
    
    constitution += f"""
## Article V: Shard Structure
- Total Shards: 71 (Monster group primes)
- Each member exists in primary shard + 7 echo shards
- Replication ensures fault tolerance and availability

## Article VI: Virtual Author Properties
Each virtual author is reconstructed from:
- Commit history (hashes, messages, diffs)
- Code patterns (feature_driven, bug_fixer, refactorer, builder, optimizer)
- Personality vector (creativity, productivity, precision, collaboration, innovation)

## Article VII: Governance
- Proposals require 51% voting power
- Execution requires 71% voting power
- Emergency actions require 90% voting power

## Article VIII: Treasury
- Initial: {dao.treasury:,} {dao.governance_token}
- Managed by DAO members
- Allocated to skill development and author rewards

∞ Ratified by 71 Shards ∞
"""
    
    Path("DAO_CONSTITUTION.md").write_text(constitution)
    
    # Statistics
    print("\n" + "=" * 70)
    print("📊 DAO Statistics:")
    print(f"  Members: {len(virtual_authors)}")
    print(f"  Total commits analyzed: {sum(v.commit_count for v in virtual_authors)}")
    print(f"  Shards occupied: {occupied_shards}/71")
    print(f"  Author instances: {total_instances}")
    
    print("\n🏆 Top 5 by Voting Power:")
    for username, power in sorted(dao.voting_power.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {username:20s}: {power:,} votes")
    
    print("\n🔮 Shard Distribution (sample):")
    for shard_id in sorted(shard_distribution.keys())[:10]:
        authors = shard_distribution[shard_id]
        if authors:
            print(f"  Shard {shard_id:2d}: {len(authors):2d} authors - {', '.join(authors[:3])}...")
    
    print(f"\n💾 Files created:")
    print(f"  - onlyskills_dao.json (DAO data)")
    print(f"  - DAO_CONSTITUTION.md (constitution)")
    
    print("\n🎯 DAO Members (Virtual Authors):")
    for author in virtual_authors:
        print(f"  - {author.username} (Shard {author.shard_id}, Prime {author.prime})")
    
    print("\n∞ DAO Created. Authors Reconstructed. 71 Shards Active. ∞")

if __name__ == "__main__":
    main()
