#!/usr/bin/env python3
"""Assign developers to Monster group structure as filters"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List

MONSTER_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]

# 10 Mathematical areas (from Monster project)
MATHEMATICAL_AREAS = [
    "K-theory / Bott periodicity",
    "Elliptic curves / CM theory",
    "Hilbert modular forms",
    "Siegel modular forms",
    "Calabi-Yau threefolds",
    "Monster moonshine",
    "Generalized moonshine",
    "Heterotic strings",
    "ADE classification",
    "Topological modular forms"
]

@dataclass
class DeveloperFilter:
    """Developer as a filter in Monster group structure"""
    username: str
    shard_id: int
    prime: int
    area: str
    
    # Filter properties
    specialization: str
    languages: List[str]
    domains: List[str]
    
    # Monster form
    monster_form: str  # How they manifest in the group
    filter_type: str   # What they filter for
    
    def to_rdf(self) -> str:
        subject = f"<https://onlyskills.com/filter/{self.username}>"
        return f"""{subject} rdf:type zkerdfa:DeveloperFilter .
{subject} zkerdfa:shardId {self.shard_id} .
{subject} zkerdfa:prime {self.prime} .
{subject} zkerdfa:area "{self.area}" .
{subject} zkerdfa:monsterForm "{self.monster_form}" .
{subject} zkerdfa:filterType "{self.filter_type}" ."""

def assign_to_monster_form(username: str, shard_id: int, skills: dict) -> DeveloperFilter:
    """Assign developer to Monster group structure"""
    
    # Map to mathematical area (7 shards per area)
    area_id = shard_id // 7
    area = MATHEMATICAL_AREAS[area_id % 10]
    
    # Determine Monster form based on skills
    languages = skills.get("languages", [])
    domains = skills.get("domains", [])
    
    # Monster form: How they manifest
    if "Rust" in languages and "Lean4" in languages:
        monster_form = "Proof-Systems Bridge"
        filter_type = "formal_verification"
    elif "Python" in languages and "ML" in domains:
        monster_form = "Neural Moonshine"
        filter_type = "machine_learning"
    elif "Haskell" in languages or "OCaml" in languages:
        monster_form = "Functional Symmetry"
        filter_type = "type_theory"
    elif "C" in languages or "C++" in languages:
        monster_form = "Systems Kernel"
        filter_type = "low_level"
    elif "JavaScript" in languages or "TypeScript" in languages:
        monster_form = "Web Manifold"
        filter_type = "frontend"
    else:
        monster_form = "General Automorphism"
        filter_type = "general_purpose"
    
    # Specialization based on area
    specializations = {
        "K-theory / Bott periodicity": "Topological Structures",
        "Elliptic curves / CM theory": "Cryptographic Primitives",
        "Hilbert modular forms": "Number Theory",
        "Siegel modular forms": "Algebraic Geometry",
        "Calabi-Yau threefolds": "String Theory",
        "Monster moonshine": "Modular Functions",
        "Generalized moonshine": "Representation Theory",
        "Heterotic strings": "Quantum Field Theory",
        "ADE classification": "Lie Algebras",
        "Topological modular forms": "Homotopy Theory"
    }
    
    return DeveloperFilter(
        username=username,
        shard_id=shard_id,
        prime=MONSTER_PRIMES[shard_id % 15],
        area=area,
        specialization=specializations[area],
        languages=languages,
        domains=domains,
        monster_form=monster_form,
        filter_type=filter_type
    )

def main():
    print("🔮 Assigning Developers to Monster Group Structure")
    print("=" * 70)
    
    # Load rankings
    rankings = json.loads(Path("github_rankings.json").read_text())
    
    # Example skills (in production, extract from GitHub)
    skills_db = {
        "torvalds": {"languages": ["C", "Assembly"], "domains": ["OS", "Kernel"]},
        "gvanrossum": {"languages": ["Python", "C"], "domains": ["Language Design"]},
        "dhh": {"languages": ["Ruby", "JavaScript"], "domains": ["Web", "Rails"]},
        "tenderlove": {"languages": ["Ruby", "C"], "domains": ["Performance"]},
        "antirez": {"languages": ["C", "Tcl"], "domains": ["Databases", "Redis"]},
        "tj": {"languages": ["JavaScript", "Go"], "domains": ["Web", "CLI"]},
        "sindresorhus": {"languages": ["JavaScript", "TypeScript"], "domains": ["npm", "Tools"]},
        "addyosmani": {"languages": ["JavaScript"], "domains": ["Web", "Performance"]},
        "paulirish": {"languages": ["JavaScript"], "domains": ["DevTools", "Web"]},
        "jeresig": {"languages": ["JavaScript"], "domains": ["jQuery", "Web"]},
    }
    
    # Assign to Monster forms
    filters = []
    for ranking in rankings[:10]:  # Real developers only
        username = ranking["username"]
        shard_id = ranking["shard_id"]
        skills = skills_db.get(username, {"languages": [], "domains": []})
        
        dev_filter = assign_to_monster_form(username, shard_id, skills)
        filters.append(dev_filter)
        
        print(f"Shard {shard_id:2d} | {username:20s} | {dev_filter.monster_form:25s} | {dev_filter.area}")
    
    # Save filters
    filters_data = [asdict(f) for f in filters]
    Path("developer_filters.json").write_text(json.dumps(filters_data, indent=2))
    
    # Generate RDF
    rdf_lines = [
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix zkerdfa: <https://onlyskills.com/zkerdfa#> .",
        "",
        "# Developer Filters - Monster Group Structure",
        ""
    ]
    for f in filters:
        rdf_lines.append(f.to_rdf())
        rdf_lines.append("")
    
    Path("developer_filters.ttl").write_text("\n".join(rdf_lines))
    
    # Statistics
    print("\n" + "=" * 70)
    print("📊 Monster Form Distribution:")
    
    forms = {}
    for f in filters:
        forms[f.monster_form] = forms.get(f.monster_form, 0) + 1
    
    for form, count in sorted(forms.items(), key=lambda x: x[1], reverse=True):
        print(f"  {form:30s}: {count} developers")
    
    print("\n📈 Area Distribution:")
    areas = {}
    for f in filters:
        areas[f.area] = areas.get(f.area, 0) + 1
    
    for area, count in sorted(areas.items(), key=lambda x: x[1], reverse=True):
        print(f"  {area:35s}: {count} developers")
    
    print("\n🔍 Filter Types:")
    filter_types = {}
    for f in filters:
        filter_types[f.filter_type] = filter_types.get(f.filter_type, 0) + 1
    
    for ftype, count in sorted(filter_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ftype:25s}: {count} developers")
    
    print(f"\n💾 Files created:")
    print(f"  - developer_filters.json (Monster forms)")
    print(f"  - developer_filters.ttl (zkERDAProlog RDF)")
    
    print("\n🔮 How Filters Work:")
    print("  - Each developer is a filter in Monster group")
    print("  - 71 shards map to 10 mathematical areas")
    print("  - Skills determine Monster form (manifestation)")
    print("  - Recruiters filter by: area, form, or type")
    
    print("\n💡 Example Queries:")
    print("  - Find all 'Proof-Systems Bridge' developers")
    print("  - Find developers in 'Elliptic curves' area")
    print("  - Find 'formal_verification' filter types")
    
    print("\n∞ 71 Shards. 10 Areas. Infinite Filters. ∞")

if __name__ == "__main__":
    main()
