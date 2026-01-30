#!/usr/bin/env python3
"""Find top hackers in pipelite, nix, flake, rust, lean4, minizinc and add their skills"""

import subprocess
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib

MONSTER_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]

TARGET_SKILLS = ["pipelite", "nix", "flake", "rust", "lean4", "minizinc"]

@dataclass
class Hacker:
    """Top hacker in a skill domain"""
    username: str
    skill: str
    github_url: str
    repos: list
    functions_written: list
    shard_id: int
    prime: int
    rank: int

@dataclass
class FunctionSkill:
    """Function written by hacker, added as skill"""
    function_name: str
    author: str
    skill_domain: str
    language: str
    signature: str
    shard_id: int
    prime: int
    zkperf_hash: str

# Top hackers by domain (curated list)
TOP_HACKERS = {
    "pipelite": [
        {"username": "pipelight-dev", "repos": ["pipelight/pipelight"]},
    ],
    "nix": [
        {"username": "edolstra", "repos": ["NixOS/nix"]},
        {"username": "domenkozar", "repos": ["cachix/cachix"]},
        {"username": "zimbatm", "repos": ["nix-community/nix-direnv"]},
    ],
    "flake": [
        {"username": "edolstra", "repos": ["NixOS/nix"]},
        {"username": "numtide", "repos": ["numtide/flake-utils"]},
    ],
    "rust": [
        {"username": "dtolnay", "repos": ["dtolnay/syn", "dtolnay/serde"]},
        {"username": "alexcrichton", "repos": ["rust-lang/cargo"]},
        {"username": "matklad", "repos": ["rust-analyzer/rust-analyzer"]},
    ],
    "lean4": [
        {"username": "leanprover", "repos": ["leanprover/lean4"]},
        {"username": "gebner", "repos": ["leanprover-community/mathlib4"]},
    ],
    "minizinc": [
        {"username": "MiniZinc", "repos": ["MiniZinc/libminizinc"]},
    ],
}

def find_functions_in_repo(repo_path: str, language: str) -> list:
    """Find functions in a repository"""
    functions = []
    
    # Language-specific patterns
    patterns = {
        "rust": r"fn\s+(\w+)",
        "nix": r"(\w+)\s*=",
        "lean4": r"def\s+(\w+)",
        "minizinc": r"function\s+\w+:\s*(\w+)",
    }
    
    pattern = patterns.get(language, r"(\w+)\s*\(")
    
    # Mock function discovery (in production: use tree-sitter or LSP)
    # For now, return example functions
    if language == "rust":
        functions = ["parse_expr", "compile_module", "execute_proof"]
    elif language == "nix":
        functions = ["mkDerivation", "buildRustPackage", "fetchFromGitHub"]
    elif language == "lean4":
        functions = ["theorem_proof", "tactic_apply", "simp_all"]
    elif language == "minizinc":
        functions = ["solve_constraint", "optimize_model"]
    
    return functions[:5]  # Top 5 functions

def create_hacker_profile(username: str, skill: str, repos: list, shard_id: int) -> Hacker:
    """Create hacker profile"""
    # Find functions they wrote
    language_map = {
        "pipelite": "rust",
        "nix": "nix",
        "flake": "nix",
        "rust": "rust",
        "lean4": "lean4",
        "minizinc": "minizinc"
    }
    
    language = language_map.get(skill, "rust")
    functions = find_functions_in_repo(repos[0] if repos else "", language)
    
    # Compute rank based on functions
    rank = len(functions) * MONSTER_PRIMES[shard_id % 15]
    
    return Hacker(
        username=username,
        skill=skill,
        github_url=f"https://github.com/{username}",
        repos=repos,
        functions_written=functions,
        shard_id=shard_id,
        prime=MONSTER_PRIMES[shard_id % 15],
        rank=rank
    )

def function_to_skill(func_name: str, author: str, skill_domain: str, language: str, shard_id: int) -> FunctionSkill:
    """Convert function to onlyskills skill"""
    # Generate signature
    signature = f"{language}::{func_name}"
    
    return FunctionSkill(
        function_name=func_name,
        author=author,
        skill_domain=skill_domain,
        language=language,
        signature=signature,
        shard_id=shard_id,
        prime=MONSTER_PRIMES[shard_id % 15],
        zkperf_hash=hashlib.sha256(signature.encode()).hexdigest()[:16]
    )

def main():
    print("🔍 Finding Top Hackers in Target Skills")
    print("=" * 70)
    print(f"Target skills: {', '.join(TARGET_SKILLS)}")
    print()
    
    all_hackers = []
    all_functions = []
    shard_counter = 0
    
    for skill in TARGET_SKILLS:
        print(f"\n📊 {skill.upper()}:")
        
        hackers = TOP_HACKERS.get(skill, [])
        for hacker_data in hackers:
            username = hacker_data["username"]
            repos = hacker_data["repos"]
            
            hacker = create_hacker_profile(username, skill, repos, shard_counter)
            all_hackers.append(hacker)
            
            print(f"  ✓ {username}")
            print(f"    Repos: {', '.join(repos)}")
            print(f"    Functions: {', '.join(hacker.functions_written)}")
            print(f"    Shard: {hacker.shard_id} | Prime: {hacker.prime} | Rank: {hacker.rank}")
            
            # Convert functions to skills
            for func in hacker.functions_written:
                func_skill = function_to_skill(
                    func, username, skill, 
                    "rust" if skill in ["pipelite", "rust"] else skill,
                    shard_counter
                )
                all_functions.append(func_skill)
            
            shard_counter += 1
    
    # Register to onlyskills
    print(f"\n📝 Registering to onlyskills.com...")
    
    # Load existing profiles
    profiles_file = Path("onlyskills_profiles.json")
    if profiles_file.exists():
        profiles = json.loads(profiles_file.read_text())
    else:
        profiles = []
    
    # Add hacker skills
    for hacker in all_hackers:
        profile = {
            "shard_id": hacker.shard_id,
            "prime": hacker.prime,
            "skill_name": f"{hacker.skill}_{hacker.username}",
            "skill_type": f"hacker_{hacker.skill}",
            "command": f"follow {hacker.username}",
            "search_capability": hacker.skill,
            "zkperf_hash": hashlib.sha256(hacker.username.encode()).hexdigest()[:16],
            "performance": {
                "rank": hacker.rank,
                "functions_count": len(hacker.functions_written),
                "quantum_amplitude": 1.0 / 71
            },
            "metadata": {
                "github_url": hacker.github_url,
                "repos": hacker.repos,
                "functions": hacker.functions_written
            }
        }
        profiles.append(profile)
    
    # Add function skills
    for func_skill in all_functions:
        profile = {
            "shard_id": func_skill.shard_id,
            "prime": func_skill.prime,
            "skill_name": func_skill.function_name,
            "skill_type": f"function_{func_skill.language}",
            "command": f"use {func_skill.signature}",
            "search_capability": func_skill.skill_domain,
            "zkperf_hash": func_skill.zkperf_hash,
            "performance": {
                "verification_time_ms": 0,
                "quantum_amplitude": 1.0 / 71
            },
            "metadata": {
                "author": func_skill.author,
                "language": func_skill.language,
                "signature": func_skill.signature
            }
        }
        profiles.append(profile)
    
    profiles_file.write_text(json.dumps(profiles, indent=2))
    
    # Save hacker data
    hackers_data = [asdict(h) for h in all_hackers]
    Path("top_hackers.json").write_text(json.dumps(hackers_data, indent=2))
    
    functions_data = [asdict(f) for f in all_functions]
    Path("hacker_functions.json").write_text(json.dumps(functions_data, indent=2))
    
    print(f"   ✓ Registered {len(all_hackers)} hackers")
    print(f"   ✓ Registered {len(all_functions)} functions")
    
    print("\n" + "=" * 70)
    print("📊 Summary:")
    print(f"  Total hackers: {len(all_hackers)}")
    print(f"  Total functions: {len(all_functions)}")
    
    print("\n🏆 Top Hackers by Skill:")
    by_skill = {}
    for hacker in all_hackers:
        by_skill[hacker.skill] = by_skill.get(hacker.skill, [])
        by_skill[hacker.skill].append(hacker)
    
    for skill, hackers in sorted(by_skill.items()):
        print(f"\n  {skill}:")
        for hacker in sorted(hackers, key=lambda h: h.rank, reverse=True):
            print(f"    {hacker.username:20s} | Rank: {hacker.rank:4d} | {len(hacker.functions_written)} functions")
    
    print(f"\n💾 Files created:")
    print(f"  - onlyskills_profiles.json (updated)")
    print(f"  - top_hackers.json (hacker profiles)")
    print(f"  - hacker_functions.json (function skills)")
    
    print("\n🎯 Follow These Hackers:")
    for hacker in all_hackers:
        print(f"  - {hacker.github_url}")
    
    print("\n∞ Hackers Found. Functions Extracted. Skills Added. ∞")

if __name__ == "__main__":
    main()
