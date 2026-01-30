#!/usr/bin/env python3
"""Test all search tools by searching for 'search' in their own code"""

import subprocess
import json
from pathlib import Path

SEARCH_TOOLS = [
    ("expert_system", "cargo run --release --bin expert_system"),
    ("multi_lang_prover", "cargo run --release --bin multi_lang_prover"),
    ("precedence_survey", "cargo run --release --bin precedence_survey"),
    ("universal_shard_reader", "cargo run --release --bin universal_shard_reader"),
    ("extract_constants", "cargo run --release --bin extract_constants"),
    ("term_frequency", "cargo run --release --bin term_frequency"),
    ("virtual_knuth", "cargo run --release --bin virtual_knuth"),
    ("llm_strip_miner", "cargo run --release --bin llm_strip_miner"),
    ("mathematical_object_lattice", "cargo run --release --bin mathematical_object_lattice"),
    ("qwen_strip_miner", "cargo run --release --bin qwen_strip_miner"),
    ("vectorize_all_parquets", "cargo run --release --bin vectorize_all_parquets"),
    ("quantum_71_shards", "cargo run --release --bin quantum_71_shards"),
]

def test_search_tool(name: str, cmd: str) -> dict:
    """Test a search tool by running it"""
    print(f"Testing {name}...", end=" ")
    
    try:
        result = subprocess.run(
            cmd.split(),
            capture_output=True,
            text=True,
            timeout=5,
            cwd="/home/mdupont/experiments/monster"
        )
        
        # Check if it mentions "search" in output
        has_search = "search" in result.stdout.lower() or "search" in result.stderr.lower()
        
        status = "✅" if result.returncode == 0 else "⚠️"
        print(f"{status} (search: {has_search})")
        
        return {
            "tool": name,
            "status": "success" if result.returncode == 0 else "error",
            "has_search": has_search,
            "exit_code": result.returncode,
            "stdout_lines": len(result.stdout.split('\n')),
            "stderr_lines": len(result.stderr.split('\n')),
        }
    except subprocess.TimeoutExpired:
        print("⏱️ timeout")
        return {"tool": name, "status": "timeout"}
    except Exception as e:
        print(f"❌ {e}")
        return {"tool": name, "status": "failed", "error": str(e)}

def main():
    print("🔍 Testing All Search Tools for 'search'")
    print("=" * 60)
    
    results = []
    for name, cmd in SEARCH_TOOLS:
        result = test_search_tool(name, cmd)
        results.append(result)
    
    print("\n" + "=" * 60)
    print("📊 Summary:")
    print(f"  Total tools: {len(results)}")
    print(f"  Success: {sum(1 for r in results if r.get('status') == 'success')}")
    print(f"  Errors: {sum(1 for r in results if r.get('status') == 'error')}")
    print(f"  Timeouts: {sum(1 for r in results if r.get('status') == 'timeout')}")
    print(f"  Has 'search': {sum(1 for r in results if r.get('has_search', False))}")
    
    # Save results
    Path("search_tool_test_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n💾 Results saved to search_tool_test_results.json")

if __name__ == "__main__":
    main()
