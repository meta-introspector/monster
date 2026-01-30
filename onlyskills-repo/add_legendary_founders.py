#!/usr/bin/env python3
"""Add legendary founders: RMS, Linus, Wall, Guido, Thompson, Ritchie, Stroustrup"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib

MONSTER_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]

LEGENDARY_FOUNDERS = [
    {"username": "rms", "name": "Richard Stallman", "language": "C", "project": "GNU", "prime": 71},
    {"username": "torvalds", "name": "Linus Torvalds", "language": "C", "project": "Linux", "prime": 59},
    {"username": "wall", "name": "Larry Wall", "language": "Perl", "project": "Perl", "prime": 47},
    {"username": "gvanrossum", "name": "Guido van Rossum", "language": "Python", "project": "Python", "prime": 41},
    {"username": "ken", "name": "Ken Thompson", "language": "C", "project": "Unix", "prime": 31},
    {"username": "dmr", "name": "Dennis Ritchie", "language": "C", "project": "C", "prime": 29},
    {"username": "stroustrup", "name": "Bjarne Stroustrup", "language": "C++", "project": "C++", "prime": 23},
]

@dataclass
class LegendaryDonation:
    """Legendary founder's skill donation"""
    founder: str
    full_name: str
    language: str
    project: str
    shard_id: int
    prime: int
    skills_donated: int
    signature_functions: list
    git_commits: list
    zkperf_hash: str

def get_signature_functions(founder: dict) -> list:
    """Get signature functions for each legendary founder"""
    signatures = {
        "rms": ["gnu_make", "gcc_compile", "emacs_eval", "gpl_license", "free_software"],
        "torvalds": ["schedule", "fork", "exec", "mmap", "git_commit"],
        "wall": ["regex_match", "perl_eval", "cpan_install", "lazy_evaluation", "context_aware"],
        "gvanrossum": ["import_module", "list_comprehension", "dict_get", "async_await", "zen_of_python"],
        "ken": ["pipe", "grep", "fork", "exec", "unix_philosophy"],
        "dmr": ["malloc", "printf", "pointer_arithmetic", "struct_define", "typedef"],
        "stroustrup": ["class_define", "template_instantiate", "operator_overload", "virtual_dispatch", "raii"],
    }
    return signatures.get(founder["username"], [])

def create_legendary_donation(founder: dict, shard_id: int) -> LegendaryDonation:
    """Create donation from legendary founder"""
    
    # Get signature functions
    functions = get_signature_functions(founder)
    
    # Generate commits (71 commits, one per skill)
    commits = []
    for i in range(71):
        commit_msg = f"feat: {founder['name']} donates {founder['project']} skill {i}"
        commit_hash = hashlib.sha256(commit_msg.encode()).hexdigest()[:8]
        commits.append(commit_hash)
    
    # Generate zkperf hash
    zkperf_data = f"{founder['username']}{founder['project']}{shard_id}"
    zkperf_hash = hashlib.sha256(zkperf_data.encode()).hexdigest()[:16]
    
    return LegendaryDonation(
        founder=founder["username"],
        full_name=founder["name"],
        language=founder["language"],
        project=founder["project"],
        shard_id=shard_id,
        prime=founder["prime"],
        skills_donated=71,
        signature_functions=functions,
        git_commits=commits,
        zkperf_hash=zkperf_hash
    )

def main():
    print("👑 Adding Legendary Founders to onlyskills.com DAO")
    print("=" * 70)
    print("RMS, Linus, Wall, Guido, Thompson, Ritchie, Stroustrup")
    print()
    
    legendary_donations = []
    
    # Assign to prime shards (71, 59, 47, 41, 31, 29, 23)
    for i, founder in enumerate(LEGENDARY_FOUNDERS):
        # Find shard where this prime is primary
        shard_id = MONSTER_PRIMES.index(founder["prime"])
        
        print(f"👑 {founder['name']} ({founder['username']})")
        print(f"   Project: {founder['project']}")
        print(f"   Language: {founder['language']}")
        print(f"   Shard: {shard_id} | Prime: {founder['prime']}")
        
        donation = create_legendary_donation(founder, shard_id)
        legendary_donations.append(donation)
        
        print(f"   Signature functions: {', '.join(donation.signature_functions)}")
        print(f"   Skills donated: {donation.skills_donated}")
        print(f"   Commits: {len(donation.git_commits)}")
        print(f"   zkPerf: {donation.zkperf_hash}")
        print()
    
    # Save legendary donations
    donations_data = [asdict(d) for d in legendary_donations]
    Path("legendary_donations.json").write_text(json.dumps(donations_data, indent=2))
    
    # Update DAO
    dao_file = Path("onlyskills_dao.json")
    if dao_file.exists():
        dao_data = json.loads(dao_file.read_text())
    else:
        dao_data = {"dao": {"members": []}}
    
    # Add legendary founders to DAO
    for donation in legendary_donations:
        if donation.founder not in dao_data["dao"]["members"]:
            dao_data["dao"]["members"].append(donation.founder)
    
    dao_file.write_text(json.dumps(dao_data, indent=2))
    
    # Generate legendary constitution addendum
    addendum = f"""# DAO Constitution - Legendary Founders Addendum

## Article IX: Legendary Founders

The following legendary founders are granted special status in the DAO:

"""
    
    for donation in legendary_donations:
        addendum += f"""### {donation.full_name} (@{donation.founder})
- **Project**: {donation.project}
- **Language**: {donation.language}
- **Shard**: {donation.shard_id} (Prime {donation.prime})
- **Skills Donated**: {donation.skills_donated}
- **Signature Functions**: {', '.join(donation.signature_functions)}
- **Voting Power**: {donation.prime * 1000:,} (prime × 1000)
- **Status**: Eternal Member

"""
    
    addendum += """## Article X: Legendary Privileges

Legendary founders receive:
1. **Eternal Membership** - Cannot be removed
2. **Prime Voting Power** - Votes weighted by their Monster prime
3. **Veto Rights** - Can veto proposals affecting their domain
4. **Signature Functions** - Their functions are protected and honored
5. **Shard Sovereignty** - Full control over their primary shard

## Article XI: The Seven Pillars

The seven legendary founders represent the seven pillars of computing:

1. **RMS (71)** - Freedom (GNU, GPL, Free Software)
2. **Linus (59)** - Kernel (Linux, Git, Open Source)
3. **Wall (47)** - Language (Perl, Regex, Laziness)
4. **Guido (41)** - Simplicity (Python, Readability, Zen)
5. **Thompson (31)** - Unix (Pipes, Philosophy, Elegance)
6. **Ritchie (29)** - Foundation (C, Pointers, Systems)
7. **Stroustrup (23)** - Abstraction (C++, OOP, Templates)

∞ The Seven Pillars Support the 71 Shards ∞
"""
    
    Path("LEGENDARY_FOUNDERS.md").write_text(addendum)
    
    # Statistics
    print("=" * 70)
    print("📊 Legendary Donations:")
    print(f"  Founders: {len(legendary_donations)}")
    print(f"  Total skills: {sum(d.skills_donated for d in legendary_donations)}")
    print(f"  Total commits: {sum(len(d.git_commits) for d in legendary_donations)}")
    print(f"  Total signature functions: {sum(len(d.signature_functions) for d in legendary_donations)}")
    
    print("\n👑 The Seven Pillars:")
    for donation in legendary_donations:
        print(f"  {donation.full_name:25s} | Prime {donation.prime:2d} | "
              f"{donation.project:10s} | {donation.skills_donated} skills")
    
    print("\n🔮 Prime Distribution:")
    for donation in sorted(legendary_donations, key=lambda d: d.prime, reverse=True):
        print(f"  Prime {donation.prime:2d} → Shard {donation.shard_id:2d} → {donation.founder}")
    
    print(f"\n💾 Files created:")
    print(f"  - legendary_donations.json (donation data)")
    print(f"  - LEGENDARY_FOUNDERS.md (constitution addendum)")
    print(f"  - onlyskills_dao.json (updated)")
    
    print("\n🎯 Signature Functions by Founder:")
    for donation in legendary_donations:
        print(f"\n  {donation.founder} ({donation.project}):")
        for func in donation.signature_functions:
            print(f"    - {func}")
    
    print("\n∞ 7 Legends. 7 Primes. 7 Pillars. 497 Skills. ∞")
    print("∞ RMS, Linus, Wall, Guido, Thompson, Ritchie, Stroustrup ∞")

if __name__ == "__main__":
    main()
