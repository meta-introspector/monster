#!/bin/bash
# Complete build, test, and prime 71 resonance analysis

set -e

echo "🔬 COMPLETE SYSTEM BUILD & PRIME 71 RESONANCE"
echo "=============================================="
echo ""

# Build everything
echo "📦 Phase 1: Build All Binaries"
echo "==============================="
cd lmfdb-rust

echo "Building rust_gcd..."
cargo build --release --bin rust_gcd 2>&1 | grep -E "(Compiling|Finished)"

echo "Building hecke_on_proof..."
cargo build --release --bin hecke_on_proof 2>&1 | grep -E "(Compiling|Finished)"

cd ..

# Run with perf and capture everything
echo ""
echo "📊 Phase 2: Capture Performance Data"
echo "====================================="

echo "Running Python with full perf..."
sudo perf record -e cycles:u,instructions:u,branches:u,cache-misses:u \
  --call-graph dwarf \
  -o perf_python.data \
  python3 test_hilbert.py 2>&1 | tail -3

echo "Running Rust with full perf..."
sudo perf record -e cycles:u,instructions:u,branches:u,cache-misses:u \
  --call-graph dwarf \
  -o perf_rust.data \
  ./lmfdb-rust/target/release/rust_gcd 2>&1 | tail -3

# Generate reports
echo ""
echo "📈 Phase 3: Generate Perf Reports"
echo "=================================="

sudo perf report -i perf_python.data --stdio > perf_python_report.txt 2>&1
sudo perf report -i perf_rust.data --stdio > perf_rust_report.txt 2>&1

echo "Python report: perf_python_report.txt"
echo "Rust report: perf_rust_report.txt"

# Analyze for prime 71
echo ""
echo "🔮 Phase 4: Prime 71 Resonance Analysis"
echo "========================================"

cat > analyze_71.py << 'PYEOF'
import re
import sys

MONSTER_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]

def analyze_file(filename):
    """Analyze file for prime 71 resonance"""
    try:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return 0.0, {}
    
    # Extract all numbers
    numbers = re.findall(r'\b\d+\b', content)
    
    resonances = {p: 0 for p in MONSTER_PRIMES}
    total = 0
    
    for num_str in numbers:
        try:
            num = int(num_str)
            if num > 0:
                total += 1
                for p in MONSTER_PRIMES:
                    if num % p == 0:
                        resonances[p] += 1
        except:
            pass
    
    if total == 0:
        return 0.0, resonances
    
    # Calculate resonance percentages
    resonance_pct = {p: (count / total) * 100 for p, count in resonances.items()}
    
    return resonance_pct.get(71, 0.0), resonance_pct

# Analyze all files
files = [
    'perf_python_report.txt',
    'perf_rust_report.txt',
    'BISIMULATION_INDEX.md',
    'BISIMULATION_SUMMARY.md',
    'COMPLETE_BISIMULATION_PROOF.md',
    'HECKE_ON_BISIMULATION.md',
    'RFC_HECKE_TRANSLATION.md',
    'lmfdb-rust/src/bin/rust_gcd.rs',
    'test_hilbert.py',
]

results = []
for filename in files:
    p71, resonances = analyze_file(filename)
    results.append((filename, p71, resonances))

# Sort by prime 71 resonance
results.sort(key=lambda x: x[1], reverse=True)

print("\n🎯 PRIME 71 RESONANCE RANKING")
print("=" * 70)
print(f"{'File':<40} {'Prime 71':<10} {'Top 3 Primes'}")
print("-" * 70)

for filename, p71, resonances in results:
    # Get top 3 primes
    top3 = sorted(resonances.items(), key=lambda x: x[1], reverse=True)[:3]
    top3_str = ", ".join([f"{p}:{v:.1f}%" for p, v in top3])
    
    print(f"{filename:<40} {p71:>6.2f}%    {top3_str}")

print("\n🔬 DETAILED PRIME 71 ANALYSIS")
print("=" * 70)

for filename, p71, resonances in results[:3]:  # Top 3 files
    if p71 > 0:
        print(f"\n{filename}:")
        print(f"  Prime 71 resonance: {p71:.2f}%")
        print(f"  All Monster primes:")
        for p in MONSTER_PRIMES:
            if resonances[p] > 0:
                print(f"    Prime {p:>2}: {resonances[p]:>6.2f}%")
PYEOF

python3 analyze_71.py

# Check actual values in perf data
echo ""
echo "🔍 Phase 5: Extract Key Metrics"
echo "================================"

echo ""
echo "Python metrics:"
grep -E "cycles|instructions|branches|cache-misses" perf_python_report.txt | head -10 || echo "  (no data)"

echo ""
echo "Rust metrics:"
grep -E "cycles|instructions|branches|cache-misses" perf_rust_report.txt | head -10 || echo "  (no data)"

# Test actual computation
echo ""
echo "🧪 Phase 6: Test with Prime 71"
echo "==============================="

cat > test_71.py << 'PYEOF'
# Test GCD with values involving 71
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Test cases involving 71
test_cases = [
    (71, 1),
    (71, 71),
    (142, 71),  # 2*71
    (213, 71),  # 3*71
    (71, 2),
    (71, 3),
    (71, 5),
]

print("Testing GCD with prime 71:")
for a, b in test_cases:
    result = gcd(a, b)
    print(f"  gcd({a:3}, {b:2}) = {result:2}")
PYEOF

python3 test_71.py

echo ""
echo "✅ ANALYSIS COMPLETE"
echo "===================="
echo ""
echo "Generated files:"
echo "  - perf_python.data (binary perf data)"
echo "  - perf_rust.data (binary perf data)"
echo "  - perf_python_report.txt (text report)"
echo "  - perf_rust_report.txt (text report)"
echo ""
echo "Prime 71 resonance analysis complete!"
