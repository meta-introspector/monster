#!/usr/bin/env python3
"""Find Monster group resonance in harmonic analysis"""
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Monster group primes
MONSTER_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]

# Monster prime powers
MONSTER_FACTORS = {
    2: 46, 3: 20, 5: 9, 7: 6, 11: 2, 13: 3,
    17: 1, 19: 1, 23: 1, 29: 1, 31: 1, 41: 1, 47: 1, 59: 1, 71: 1
}

def find_resonance(harmonics_file, output_file):
    print(f"👹 Loading harmonics from: {harmonics_file}")
    
    # Load harmonics (JSON from Julia)
    json_file = harmonics_file + ".json" if not harmonics_file.endswith('.json') else harmonics_file
    with open(json_file) as f:
        data = json.load(f)
    
    results = []
    
    for entry in data:
        reg = entry['register']
        top_freqs = entry['top_frequencies']
        top_powers = entry['top_powers']
        
        # Check divisibility by Monster primes
        divisibility = {}
        for prime in MONSTER_PRIMES:
            count = sum(1 for freq in top_freqs if freq % prime == 0)
            divisibility[f'div_{prime}'] = count
            divisibility[f'div_{prime}_pct'] = (count / len(top_freqs) * 100) if top_freqs else 0
        
        # Compute resonance score
        resonance_score = 0
        for prime, power in MONSTER_FACTORS.items():
            # Weight by prime power in Monster factorization
            div_pct = divisibility[f'div_{prime}_pct']
            resonance_score += div_pct * power
        
        # Normalize by total power
        resonance_score /= sum(MONSTER_FACTORS.values())
        
        results.append({
            'register': reg,
            'count': entry['count'],
            'mean': entry['mean'],
            'std': entry['std'],
            'fft_size': entry['fft_size'],
            'total_power': entry['total_power'],
            'resonance_score': resonance_score,
            **divisibility
        })
        
        print(f"  ✓ {reg}: resonance score = {resonance_score:.2f}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by resonance score
    df = df.sort_values('resonance_score', ascending=False)
    
    # Save to parquet
    df.to_parquet(output_file, index=False)
    
    print(f"\n✅ Monster resonance analysis complete!")
    print(f"\n🏆 Top 3 resonant registers:")
    for i, row in df.head(3).iterrows():
        print(f"   {i+1}. {row['register']}: {row['resonance_score']:.2f}")
    
    print(f"\n💾 Saved to: {output_file}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: monster_resonance.py <harmonics.parquet> <output.parquet>")
        sys.exit(1)
    
    find_resonance(sys.argv[1], sys.argv[2])
