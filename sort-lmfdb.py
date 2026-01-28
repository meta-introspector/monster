#!/usr/bin/env python3
"""
Sort LMFDB mathematical objects into 71 Monster shards by prime resonance
"""

import json
import os
from pathlib import Path
from collections import defaultdict

MONSTER_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]

def calculate_prime_resonance(obj):
    """Calculate which prime(s) an object resonates with"""
    resonances = []
    
    # Extract numerical properties
    numbers = extract_numbers(obj)
    
    for prime in MONSTER_PRIMES:
        score = 0
        for num in numbers:
            if num % prime == 0:
                score += 1
        
        if score > 0:
            resonances.append((prime, score / len(numbers) if numbers else 0))
    
    return resonances

def extract_numbers(obj):
    """Extract all numerical values from an object"""
    numbers = []
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            # Check for conductor, level, degree, rank, etc.
            if isinstance(value, int) and value > 0:
                numbers.append(value)
            elif isinstance(value, list):
                numbers.extend([v for v in value if isinstance(v, int) and v > 0])
            elif isinstance(value, dict):
                numbers.extend(extract_numbers(value))
    
    return numbers

def assign_to_shard(obj):
    """Assign object to primary shard based on strongest resonance"""
    resonances = calculate_prime_resonance(obj)
    
    if not resonances:
        return 1  # Default to shard 1
    
    # Sort by resonance score
    resonances.sort(key=lambda x: x[1], reverse=True)
    
    return resonances[0][0]  # Return prime with highest resonance

def process_lmfdb_file(filepath, output_dir):
    """Process a single LMFDB JSON file"""
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Initialize shard buckets
    shards = defaultdict(list)
    
    # Sort objects into shards
    if isinstance(data, list):
        for obj in data:
            shard = assign_to_shard(obj)
            shards[shard].append(obj)
    elif isinstance(data, dict):
        for key, obj in data.items():
            shard = assign_to_shard(obj)
            obj['_id'] = key
            shards[shard].append(obj)
    
    # Write to shard files
    filename = Path(filepath).stem
    for shard_num, objects in shards.items():
        shard_dir = output_dir / f"shard-{shard_num:02d}" / "data" / "lmfdb"
        shard_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = shard_dir / f"{filename}.json"
        with open(output_file, 'w') as f:
            json.dump(objects, f, indent=2)
        
        print(f"  Shard {shard_num}: {len(objects)} objects → {output_file}")
    
    return shards

def main():
    lmfdb_dir = Path("/mnt/data1/meta-introspector/data/lmfdb-collected")
    output_dir = Path("/home/mdupont/experiments/monster/monster-shards")
    
    print("🔷 Sorting LMFDB by Prime Resonance")
    print("=" * 50)
    print()
    
    total_stats = defaultdict(int)
    
    # Process each LMFDB file
    for json_file in lmfdb_dir.glob("*.json"):
        shards = process_lmfdb_file(json_file, output_dir)
        
        for shard_num, objects in shards.items():
            total_stats[shard_num] += len(objects)
        
        print()
    
    # Print summary
    print("=" * 50)
    print("Summary:")
    print()
    
    for shard_num in sorted(total_stats.keys()):
        is_prime = shard_num in MONSTER_PRIMES
        marker = "★" if is_prime else " "
        print(f"  Shard {shard_num:2d} {marker}: {total_stats[shard_num]:4d} objects")
    
    print()
    print(f"Total: {sum(total_stats.values())} objects distributed across {len(total_stats)} shards")
    print()
    print("✅ LMFDB sorted by Monster prime resonance!")

if __name__ == "__main__":
    main()
