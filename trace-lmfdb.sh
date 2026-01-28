#!/bin/bash
# Run LMFDB Python code, trace with perf, reconstruct Rust

echo "🔬 LMFDB Trace-Driven Reconstruction"
echo "===================================="
echo ""

LMFDB_PATH="/mnt/data1/nix/source/github/meta-introspector/lmfdb"
OUTPUT_DIR="perf_traces/lmfdb"

mkdir -p "$OUTPUT_DIR"

# Test script to run
cat > test_hilbert.py << 'EOF'
# Simple number theory operations - no LMFDB dependencies
print("Testing number theory operations...")

# Simulate Hilbert field arithmetic
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    gcd_val, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd_val, x, y

# Run computations
results = []
for i in range(1000):
    a = 2**i % 71  # Monster prime!
    b = 3**i % 71
    g = gcd(a, b)
    results.append(g)

print(f"Completed {len(results)} GCD computations")
print(f"Sample results: {results[:10]}")
print("Success!")
EOF

echo "Phase 1: Run Python with perf"
echo "------------------------------"

sudo perf record \
    -e cycles:u \
    --call-graph dwarf \
    --intr-regs=AX,BX,CX,DX,SI,DI,R8,R9,R10,R11,R12,R13,R14,R15 \
    -o "$OUTPUT_DIR/hilbert.data" \
    -- python3 test_hilbert.py

echo ""
echo "Phase 2: Generate perf script"
echo "------------------------------"

sudo perf script -i "$OUTPUT_DIR/hilbert.data" > "$OUTPUT_DIR/hilbert.script"

echo "  ✓ Saved to $OUTPUT_DIR/hilbert.script"
echo ""

echo "Phase 3: Analyze traces"
echo "-----------------------"

# Extract function calls
echo "Top functions called:"
grep -E "^\s+[0-9a-f]+" "$OUTPUT_DIR/hilbert.script" | \
    awk '{print $NF}' | \
    sort | uniq -c | sort -rn | head -20

echo ""
echo "Phase 4: Extract register patterns"
echo "-----------------------------------"

# Extract register values
grep "AX:" "$OUTPUT_DIR/hilbert.script" | head -10

echo ""
echo "✅ Traces captured!"
echo "   Data: $OUTPUT_DIR/hilbert.data"
echo "   Script: $OUTPUT_DIR/hilbert.script"
echo ""
echo "Next: Analyze traces to reconstruct Rust code"
