#!/usr/bin/env python3
"""Ingest consumed repos into onlyskills.com and register as AI skills"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib

MONSTER_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]

@dataclass
class AISkill:
    """Skill that can be added to AI agent"""
    skill_id: str
    name: str
    description: str
    shard_id: int
    prime: int
    skill_type: str
    command: str
    source_repo: str
    capabilities: list
    zkperf_hash: str

def extract_skills_from_repo(repo_data: dict) -> list:
    """Extract AI skills from consumed repository"""
    skills = []
    name = repo_data["name"]
    shard_id = repo_data["shard_id"]
    
    # zkOS skills
    if name == "zos-server":
        skills.extend([
            AISkill(
                skill_id="zkwasm_loader",
                name="zkWASM Proof Loader",
                description="Load and execute eRDF zkWASM proofs with zero-knowledge verification",
                shard_id=shard_id,
                prime=repo_data["prime"],
                skill_type="proof_verification",
                command="zkos load-proof",
                source_repo="zos-server",
                capabilities=["zkwasm", "erdf", "proof_execution", "zero_knowledge"],
                zkperf_hash=hashlib.sha256(b"zkwasm_loader").hexdigest()[:16]
            ),
            AISkill(
                skill_id="evolution_server",
                name="Evolution Server",
                description="Self-evolving code generation and compilation server",
                shard_id=shard_id + 1,
                prime=MONSTER_PRIMES[(shard_id + 1) % 15],
                skill_type="code_generation",
                command="zkos evolve",
                source_repo="zos-server",
                capabilities=["code_evolution", "self_modification", "compilation"],
                zkperf_hash=hashlib.sha256(b"evolution_server").hexdigest()[:16]
            ),
            AISkill(
                skill_id="zos_71_shards",
                name="71-Shard System",
                description="Distribute computation across 71 Monster group shards",
                shard_id=shard_id + 2,
                prime=MONSTER_PRIMES[(shard_id + 2) % 15],
                skill_type="distributed_compute",
                command="zkos shard",
                source_repo="zos-server",
                capabilities=["sharding", "monster_group", "parallel_compute"],
                zkperf_hash=hashlib.sha256(b"zos_71_shards").hexdigest()[:16]
            ),
        ])
    
    # Meta-introspector skills
    elif name == "meta-introspector":
        skills.extend([
            AISkill(
                skill_id="solfunmeme",
                name="SOLFUNMEME Integration",
                description="Solana meme coin with Monster NFT pairing and MaaS form",
                shard_id=shard_id,
                prime=repo_data["prime"],
                skill_type="blockchain",
                command="meta solfunmeme",
                source_repo="meta-introspector",
                capabilities=["solana", "nft", "meme_coin", "maas_form"],
                zkperf_hash=hashlib.sha256(b"solfunmeme").hexdigest()[:16]
            ),
            AISkill(
                skill_id="dao_bootstrap",
                name="Recursive DAO Bootstrap",
                description="Self-bootstrapping DAO with event-driven blockchain",
                shard_id=shard_id + 1,
                prime=MONSTER_PRIMES[(shard_id + 1) % 15],
                skill_type="governance",
                command="meta dao-bootstrap",
                source_repo="meta-introspector",
                capabilities=["dao", "governance", "recursive_bootstrap", "blockchain"],
                zkperf_hash=hashlib.sha256(b"dao_bootstrap").hexdigest()[:16]
            ),
            AISkill(
                skill_id="grammar_constrained_llm",
                name="Grammar-Constrained LLM",
                description="LLM with formal grammar constraints for verified output",
                shard_id=shard_id + 2,
                prime=MONSTER_PRIMES[(shard_id + 2) % 15],
                skill_type="llm",
                command="meta grammar-llm",
                source_repo="meta-introspector",
                capabilities=["llm", "grammar", "verification", "constrained_generation"],
                zkperf_hash=hashlib.sha256(b"grammar_llm").hexdigest()[:16]
            ),
        ])
    
    # Zombie driver skills
    elif name == "zombie_driver2":
        skills.extend([
            AISkill(
                skill_id="automorphic_orbit_tracer",
                name="Automorphic Orbit Tracer",
                description="Trace execution orbits and detect automorphic patterns",
                shard_id=shard_id,
                prime=repo_data["prime"],
                skill_type="execution_analysis",
                command="zombie trace-orbit",
                source_repo="zombie_driver2",
                capabilities=["orbit_tracing", "automorphic", "execution_analysis"],
                zkperf_hash=hashlib.sha256(b"orbit_tracer").hexdigest()[:16]
            ),
            AISkill(
                skill_id="abi_signature_extractor",
                name="ABI Signature Extractor",
                description="Extract and analyze ABI signatures from binaries",
                shard_id=shard_id + 1,
                prime=MONSTER_PRIMES[(shard_id + 1) % 15],
                skill_type="binary_analysis",
                command="zombie extract-abi",
                source_repo="zombie_driver2",
                capabilities=["abi", "signature_extraction", "binary_analysis"],
                zkperf_hash=hashlib.sha256(b"abi_extractor").hexdigest()[:16]
            ),
        ])
    
    return skills

def register_skills_to_onlyskills(skills: list):
    """Register skills to onlyskills.com registry"""
    # Load existing profiles
    profiles_file = Path("onlyskills_profiles.json")
    if profiles_file.exists():
        profiles = json.loads(profiles_file.read_text())
    else:
        profiles = []
    
    # Add new skills
    for skill in skills:
        profile = {
            "shard_id": skill.shard_id,
            "prime": skill.prime,
            "skill_name": skill.skill_id,
            "skill_type": f"ai_{skill.skill_type}",
            "command": skill.command,
            "search_capability": skill.skill_type,
            "zkperf_hash": skill.zkperf_hash,
            "performance": {
                "verification_time_ms": 0,
                "lines_of_code": 0,
                "quantum_amplitude": 1.0 / 71
            },
            "metadata": {
                "description": skill.description,
                "source_repo": skill.source_repo,
                "capabilities": skill.capabilities
            }
        }
        profiles.append(profile)
    
    # Save updated profiles
    profiles_file.write_text(json.dumps(profiles, indent=2))
    
    return len(skills)

def generate_kiro_tool_manifest(skills: list):
    """Generate Kiro CLI tool manifest for AI to use these skills"""
    manifest = {
        "name": "monster-ai-skills",
        "version": "1.0.0",
        "description": "AI skills from zos-server, meta-introspector, zombie_driver2",
        "tools": []
    }
    
    for skill in skills:
        tool = {
            "name": skill.skill_id,
            "description": skill.description,
            "command": skill.command,
            "category": skill.skill_type,
            "inputs": [],
            "outputs": ["result"],
            "capabilities": skill.capabilities,
            "shard_id": skill.shard_id,
            "prime": skill.prime
        }
        manifest["tools"].append(tool)
    
    Path(".kiro/tools/monster-ai-skills.json").write_text(json.dumps(manifest, indent=2))
    
    return manifest

def main():
    print("🔮 Ingesting Repos into onlyskills.com")
    print("=" * 70)
    
    # Load consumed repos
    consumed = json.loads(Path("consumed_repos.json").read_text())
    
    all_skills = []
    
    for repo in consumed["consumed_repos"]:
        print(f"\n📦 Extracting skills from {repo['name']}...")
        skills = extract_skills_from_repo(repo)
        all_skills.extend(skills)
        
        for skill in skills:
            print(f"   ✓ {skill.name}")
            print(f"     Shard: {skill.shard_id} | Prime: {skill.prime}")
            print(f"     Type: {skill.skill_type}")
            print(f"     Capabilities: {', '.join(skill.capabilities)}")
    
    # Register to onlyskills
    print(f"\n📝 Registering {len(all_skills)} skills to onlyskills.com...")
    registered = register_skills_to_onlyskills(all_skills)
    print(f"   ✓ Registered {registered} skills")
    
    # Generate Kiro manifest
    print(f"\n🔧 Generating Kiro CLI tool manifest...")
    manifest = generate_kiro_tool_manifest(all_skills)
    print(f"   ✓ Created manifest with {len(manifest['tools'])} tools")
    
    # Save skills list
    skills_data = [asdict(s) for s in all_skills]
    Path("ai_skills.json").write_text(json.dumps(skills_data, indent=2))
    
    print("\n" + "=" * 70)
    print("📊 Summary:")
    print(f"  Total skills extracted: {len(all_skills)}")
    print(f"  Registered to onlyskills: {registered}")
    print(f"  Kiro tools created: {len(manifest['tools'])}")
    
    print("\n🎯 Skills by Type:")
    by_type = {}
    for skill in all_skills:
        by_type[skill.skill_type] = by_type.get(skill.skill_type, 0) + 1
    
    for skill_type, count in sorted(by_type.items()):
        print(f"  {skill_type:25s}: {count} skills")
    
    print(f"\n💾 Files created:")
    print(f"  - onlyskills_profiles.json (updated)")
    print(f"  - .kiro/tools/monster-ai-skills.json (Kiro manifest)")
    print(f"  - ai_skills.json (skills data)")
    
    print("\n🤖 AI Can Now Use These Skills:")
    for skill in all_skills:
        print(f"  - {skill.name}: {skill.command}")
    
    print("\n∞ Skills Ingested. AI Enhanced. Ready. ∞")

if __name__ == "__main__":
    main()
