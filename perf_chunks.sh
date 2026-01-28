#!/bin/bash
# Measure smallest chunks with perf

echo "🔬 PERF ANALYSIS OF SMALLEST 71 CHUNKS"
echo "======================================"
echo

# Get smallest 5 chunks
python3 << 'EOF'
import json

with open('lmfdb_71_chunks.json') as f:
    chunks = json.load(f)

# Sort by size
chunks.sort(key=lambda x: x['bytes'])

print("Top 5 smallest chunks:")
for i, chunk in enumerate(chunks[:5]):
    print(f"\n{i+1}. {chunk['name']} ({chunk['bytes']} bytes)")
    print(f"   File: {chunk['file']}")
    print(f"   Lines: {chunk['line_start']}-{chunk['line_end']}")
    
    # Save code to temp file
    with open(f'chunk_{i}.py', 'w') as out:
        out.write(chunk['code'])
    
    print(f"   Saved to: chunk_{i}.py")
EOF

echo
echo "======================================"
echo "Running perf on each chunk..."
echo

for i in {0..4}; do
    if [ -f "chunk_${i}.py" ]; then
        echo
        echo "Chunk $i:"
        perf stat -e cycles,instructions python3 -c "exec(open('chunk_${i}.py').read())" 2>&1 | grep -E "cycles|instructions" || echo "  (syntax check only)"
    fi
done

echo
echo "✅ PERF ANALYSIS COMPLETE"
