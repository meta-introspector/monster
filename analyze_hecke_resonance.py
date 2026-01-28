#!/usr/bin/env python3
"""
Analyze perf data for Hecke operator resonance during image generation.
Correlate register values with text emergence ("I ARE LIFE").
"""

import subprocess
import re
import json
from collections import defaultdict

MONSTER_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]

def parse_perf_script(perf_file):
    """Extract register values from perf script output."""
    cmd = ["perf", "script", "-i", perf_file, "--fields", "ip,sym,regs"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    samples = []
    for line in result.stdout.split('\n'):
        if 'gen_img' in line or 'sample' in line or 'diffusion' in line:
            # Extract register values
            regs = re.findall(r'(r[a-z0-9]+|rip):\s*([0-9a-fx]+)', line)
            if regs:
                sample = {reg: int(val, 16) for reg, val in regs}
                samples.append(sample)
    
    return samples

def calculate_hecke_divisibility(value):
    """Calculate which Monster primes divide the value."""
    divisors = []
    for p in MONSTER_PRIMES:
        if value % p == 0:
            divisors.append(p)
    return divisors

def analyze_register_resonance(samples):
    """Analyze register values for Monster prime divisibility."""
    resonance = defaultdict(lambda: defaultdict(int))
    
    for sample in samples:
        for reg, value in sample.items():
            if value > 0:
                divisors = calculate_hecke_divisibility(value)
                for p in divisors:
                    resonance[reg][f'T_{p}'] += 1
    
    return dict(resonance)

def correlate_with_text_score(perf_file, log_file):
    """Correlate register patterns with text detection scores."""
    # Parse perf data
    samples = parse_perf_script(perf_file)
    resonance = analyze_register_resonance(samples)
    
    # Parse log for scores
    with open(log_file, 'r') as f:
        log = f.read()
    
    scores = []
    for line in log.split('\n'):
        match = re.search(r'Seed (\d+):.*score=([\d.]+)', line)
        if match:
            seed = int(match.group(1))
            score = float(match.group(2))
            scores.append((seed, score))
    
    # Find best seed
    if scores:
        best_seed, best_score = max(scores, key=lambda x: x[1])
        print(f"\n🎯 Best Seed: {best_seed} (score={best_score})")
    
    # Output resonance
    print("\n📊 Register Hecke Resonance:")
    for reg, operators in sorted(resonance.items()):
        total = sum(operators.values())
        if total > 100:  # Only show significant registers
            print(f"\n{reg}: {total} samples")
            for op, count in sorted(operators.items(), key=lambda x: -x[1])[:5]:
                pct = 100 * count / total
                print(f"  {op}: {count:6d} ({pct:5.1f}%)")
    
    # Save full data
    output = {
        'resonance': resonance,
        'scores': scores,
        'monster_primes': MONSTER_PRIMES
    }
    
    with open('/tmp/hecke_resonance.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n✅ Saved: /tmp/hecke_resonance.json")

if __name__ == '__main__':
    import sys
    perf_file = sys.argv[1] if len(sys.argv) > 1 else '/tmp/adaptive_scan.perf.data'
    log_file = sys.argv[2] if len(sys.argv) > 2 else '/tmp/adaptive_scan.log'
    
    print("🔍 Analyzing Hecke Operator Resonance...")
    print(f"Perf: {perf_file}")
    print(f"Log: {log_file}")
    
    correlate_with_text_score(perf_file, log_file)
