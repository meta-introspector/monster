#!/usr/bin/env bash
# Complete LMFDB Hecke Analysis Pipeline
# 1. Build database with Nix
# 2. Trace all execution
# 3. Apply Hecke operators at all levels

set -e

LMFDB_PATH="/mnt/data1/nix/source/github/meta-introspector/lmfdb"
OUTPUT_DIR="$PWD/lmfdb_hecke_analysis"

echo "🔮 COMPLETE LMFDB HECKE ANALYSIS PIPELINE"
echo "=========================================="
echo ""

# Phase 1: Setup
echo "Phase 1: Setup Output Directory"
echo "--------------------------------"
mkdir -p "$OUTPUT_DIR"/{source,ast,bytecode,perf,database}
echo "✓ Created: $OUTPUT_DIR"
echo ""

# Phase 2: Source Analysis
echo "Phase 2: Analyze Source Files"
echo "------------------------------"
python3 << 'PYEOF'
import os
import json
from pathlib import Path

LMFDB_PATH = "/mnt/data1/nix/source/github/meta-introspector/lmfdb"
OUTPUT_DIR = "lmfdb_hecke_analysis"
MONSTER_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]

def hecke_resonance(value):
    """Find Monster prime with highest divisibility"""
    if value == 0:
        return 1
    resonances = {}
    for p in MONSTER_PRIMES:
        if value % p == 0:
            count = 0
            temp = abs(value)
            while temp % p == 0:
                count += 1
                temp //= p
            resonances[p] = count
    return max(resonances.items(), key=lambda x: x[1])[0] if resonances else 1

# Find all Python files
py_files = list(Path(LMFDB_PATH).rglob("*.py"))
print(f"Found {len(py_files)} Python files")

# Analyze each file
file_stats = []
for py_file in py_files[:10]:  # Start with first 10
    try:
        with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Count lines
        lines = len(content.split('\n'))
        
        # Find literal 71
        count_71 = content.count('71')
        
        # Hecke resonance of line count
        shard = hecke_resonance(lines)
        
        file_stats.append({
            'path': str(py_file.relative_to(LMFDB_PATH)),
            'lines': lines,
            'literal_71': count_71,
            'shard': shard
        })
        
    except Exception as e:
        print(f"Error reading {py_file}: {e}")

# Save results
with open(f"{OUTPUT_DIR}/source/file_analysis.json", 'w') as f:
    json.dump(file_stats, f, indent=2)

print(f"✓ Analyzed {len(file_stats)} files")
print(f"  Files with literal 71: {sum(1 for f in file_stats if f['literal_71'] > 0)}")
PYEOF

echo ""

# Phase 3: AST Analysis
echo "Phase 3: AST Analysis"
echo "---------------------"
python3 << 'PYEOF'
import ast
import json
from pathlib import Path

LMFDB_PATH = "/mnt/data1/nix/source/github/meta-introspector/lmfdb"
OUTPUT_DIR = "lmfdb_hecke_analysis"

py_files = list(Path(LMFDB_PATH).rglob("*.py"))[:10]

ast_stats = []
for py_file in py_files:
    try:
        with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
        
        tree = ast.parse(code)
        
        # Count nodes
        total_nodes = len(list(ast.walk(tree)))
        
        # Find literal 71 in AST
        count_71 = sum(1 for node in ast.walk(tree)
                      if isinstance(node, ast.Constant) and node.value == 71)
        
        ast_stats.append({
            'path': str(py_file.relative_to(LMFDB_PATH)),
            'nodes': total_nodes,
            'literal_71': count_71
        })
        
    except Exception as e:
        pass

with open(f"{OUTPUT_DIR}/ast/ast_analysis.json", 'w') as f:
    json.dump(ast_stats, f, indent=2)

print(f"✓ Analyzed {len(ast_stats)} AST trees")
print(f"  Files with AST Constant(71): {sum(1 for f in ast_stats if f['literal_71'] > 0)}")
PYEOF

echo ""

# Phase 4: Database Setup (placeholder - needs actual LMFDB setup)
echo "Phase 4: Database Setup"
echo "-----------------------"
cat > "$OUTPUT_DIR/database/setup.sh" << 'DBEOF'
#!/bin/bash
# LMFDB Database Setup
# This would normally:
# 1. Start PostgreSQL with Nix
# 2. Initialize LMFDB schema
# 3. Load data
# 4. Run queries

echo "Database setup (placeholder)"
echo "Would execute:"
echo "  nix-shell -p postgresql"
echo "  initdb -D lmfdb_data"
echo "  pg_ctl -D lmfdb_data start"
echo "  psql -f lmfdb/schema.sql"
DBEOF
chmod +x "$OUTPUT_DIR/database/setup.sh"
echo "✓ Created database setup script"
echo ""

# Phase 5: Summary
echo "Phase 5: Generate Summary"
echo "-------------------------"
python3 << 'PYEOF'
import json
from pathlib import Path

OUTPUT_DIR = "lmfdb_hecke_analysis"

# Load results
with open(f"{OUTPUT_DIR}/source/file_analysis.json") as f:
    source_stats = json.load(f)

with open(f"{OUTPUT_DIR}/ast/ast_analysis.json") as f:
    ast_stats = json.load(f)

# Summary
summary = {
    'total_files': len(source_stats),
    'total_lines': sum(f['lines'] for f in source_stats),
    'files_with_71': sum(1 for f in source_stats if f['literal_71'] > 0),
    'total_ast_nodes': sum(f['nodes'] for f in ast_stats),
    'ast_with_71': sum(1 for f in ast_stats if f['literal_71'] > 0),
}

with open(f"{OUTPUT_DIR}/summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("\n📊 SUMMARY")
print("=" * 50)
print(f"Files analyzed:        {summary['total_files']}")
print(f"Total lines:           {summary['total_lines']:,}")
print(f"Files with literal 71: {summary['files_with_71']}")
print(f"Total AST nodes:       {summary['total_ast_nodes']:,}")
print(f"AST with Constant(71): {summary['ast_with_71']}")
PYEOF

echo ""
echo "✅ PHASE 1 COMPLETE"
echo "==================="
echo ""
echo "Next steps:"
echo "1. Expand to all 356 Python files"
echo "2. Add bytecode analysis"
echo "3. Add performance tracing"
echo "4. Setup actual database"
echo ""
echo "Output directory: $OUTPUT_DIR"
