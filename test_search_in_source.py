#!/usr/bin/env python3
"""Test search tools by searching their source code for 'search'"""

import subprocess
from pathlib import Path

def search_in_source(tool_name: str) -> dict:
    """Search for 'search' in tool source code"""
    source_file = Path(f"src/bin/{tool_name}.rs")
    
    if not source_file.exists():
        return {"tool": tool_name, "status": "not_found"}
    
    content = source_file.read_text()
    
    # Count occurrences
    search_count = content.lower().count("search")
    fn_search = content.count("fn search") + content.count("search(")
    
    # Find search functions
    search_fns = []
    for line in content.split('\n'):
        if 'fn search' in line or 'fn.*search' in line:
            search_fns.append(line.strip())
    
    return {
        "tool": tool_name,
        "status": "found",
        "search_count": search_count,
        "search_functions": len(search_fns),
        "functions": search_fns[:5],  # First 5
        "has_search": search_count > 0
    }

SEARCH_TOOLS = [
    "expert_system",
    "multi_lang_prover",
    "precedence_survey",
    "universal_shard_reader",
    "extract_constants",
    "term_frequency",
    "virtual_knuth",
    "llm_strip_miner",
    "mathematical_object_lattice",
    "qwen_strip_miner",
    "vectorize_all_parquets",
    "quantum_71_shards",
]

def main():
    print("🔍 Searching for 'search' in Search Tool Source Code")
    print("=" * 70)
    
    results = []
    for tool in SEARCH_TOOLS:
        result = search_in_source(tool)
        results.append(result)
        
        if result["status"] == "found":
            print(f"{tool:30s} | search: {result['search_count']:3d} | "
                  f"fns: {result['search_functions']:2d} | "
                  f"{'✅' if result['has_search'] else '❌'}")
        else:
            print(f"{tool:30s} | ❌ not found")
    
    print("\n" + "=" * 70)
    print("📊 Summary:")
    found = [r for r in results if r["status"] == "found"]
    print(f"  Tools found: {len(found)}/{len(SEARCH_TOOLS)}")
    print(f"  Tools with 'search': {sum(1 for r in found if r['has_search'])}")
    print(f"  Total 'search' occurrences: {sum(r.get('search_count', 0) for r in found)}")
    print(f"  Total search functions: {sum(r.get('search_functions', 0) for r in found)}")
    
    # Top 5 tools by search count
    print("\n🏆 Top 5 Tools by 'search' Count:")
    top5 = sorted(found, key=lambda r: r.get('search_count', 0), reverse=True)[:5]
    for i, r in enumerate(top5, 1):
        print(f"  {i}. {r['tool']:30s} - {r['search_count']} occurrences")
    
    # Tools with search functions
    print("\n🔧 Tools with Search Functions:")
    with_fns = [r for r in found if r.get('search_functions', 0) > 0]
    for r in with_fns:
        print(f"  {r['tool']:30s} - {r['search_functions']} functions")
        for fn in r.get('functions', []):
            print(f"    {fn}")

if __name__ == "__main__":
    main()
