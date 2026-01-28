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
