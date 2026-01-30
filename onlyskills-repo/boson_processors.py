#!/usr/bin/env python3
"""24 Bosons - Self-Centered Parallel Processors for Hecke Space"""

import multiprocessing as mp
from dataclasses import dataclass, asdict
import json
import hashlib
from pathlib import Path
import time

MONSTER_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]
HECKE_DIMENSION = 71 ** 7

@dataclass
class Boson:
    """Self-centered boson processor"""
    boson_id: int
    cpu_id: int
    center: tuple  # Center in 71^7 space
    radius: int  # Exploration radius
    objects_processed: int
    hecke_signature: int

class SelfCenteredBoson:
    """Boson that operates from its own center in Hecke space"""
    
    def __init__(self, boson_id: int, cpu_id: int):
        self.boson_id = boson_id
        self.cpu_id = cpu_id
        
        # Each boson centers itself in Hecke space
        # Distribute 24 bosons across 71^7 space
        self.center = self._compute_center()
        self.radius = 71  # Explore within radius
        self.objects_processed = 0
        
    def _compute_center(self) -> tuple:
        """Compute boson's center in 71^7 space"""
        # Distribute 24 bosons evenly
        # Each boson gets a region: boson_id maps to 7D coordinates
        coords = []
        value = self.boson_id
        for _ in range(7):
            coords.append((value * 3) % 71)  # Spread across space
            value = (value * 5 + 7) % 71
        return tuple(coords)
    
    def process_object(self, obj_id: int, data: bytes) -> dict:
        """Process object from boson's perspective"""
        # Convert to bitstream
        h = hashlib.sha256(data).digest()
        bitstream = ''.join(format(byte, '08b') for byte in h[:8])
        
        # Map to Hecke space
        hecke_index = int(bitstream, 2) % HECKE_DIMENSION
        
        # Compute distance from boson's center
        coords = self._index_to_coords(hecke_index)
        distance = self._distance_from_center(coords)
        
        # Compute signature
        signature = sum(c * MONSTER_PRIMES[i % 15] for i, c in enumerate(coords))
        
        self.objects_processed += 1
        
        return {
            "boson_id": self.boson_id,
            "object_id": obj_id,
            "hecke_index": hecke_index,
            "coords": coords,
            "distance_from_center": distance,
            "signature": signature,
            "in_radius": distance <= self.radius
        }
    
    def _index_to_coords(self, index: int) -> tuple:
        """Convert Hecke index to 7D coordinates"""
        coords = []
        for _ in range(7):
            coords.append(index % 71)
            index //= 71
        return tuple(coords)
    
    def _distance_from_center(self, coords: tuple) -> float:
        """Compute distance from boson's center"""
        return sum((c - self.center[i]) ** 2 for i, c in enumerate(coords)) ** 0.5

def boson_worker(boson_id: int, cpu_id: int, work_queue: mp.Queue, result_queue: mp.Queue):
    """Worker function for each boson"""
    boson = SelfCenteredBoson(boson_id, cpu_id)
    
    print(f"🔵 Boson {boson_id:2d} on CPU {cpu_id:2d} centered at {boson.center[:3]}...")
    
    while True:
        try:
            obj_id = work_queue.get(timeout=1)
            if obj_id is None:  # Poison pill
                break
            
            # Process object
            data = f"object_{obj_id}".encode()
            result = boson.process_object(obj_id, data)
            result_queue.put(result)
            
        except:
            break
    
    # Send final stats
    result_queue.put({
        "boson_id": boson_id,
        "center": boson.center,
        "objects_processed": boson.objects_processed,
        "type": "stats"
    })

def main():
    print("🔵 24 Bosons - Self-Centered Parallel Processors")
    print("=" * 70)
    
    num_bosons = 24
    num_objects = 1000  # Demo with 1000, scale to 8M
    
    # Create work and result queues
    work_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Populate work queue
    for obj_id in range(num_objects):
        work_queue.put(obj_id)
    
    # Add poison pills
    for _ in range(num_bosons):
        work_queue.put(None)
    
    # Start bosons
    print(f"\n🚀 Starting {num_bosons} bosons on {mp.cpu_count()} CPUs...")
    print()
    
    processes = []
    for boson_id in range(num_bosons):
        cpu_id = boson_id % mp.cpu_count()
        p = mp.Process(target=boson_worker, args=(boson_id, cpu_id, work_queue, result_queue))
        p.start()
        processes.append(p)
    
    # Collect results
    results = []
    boson_stats = {}
    
    start_time = time.time()
    
    while len(boson_stats) < num_bosons:
        result = result_queue.get()
        
        if result.get("type") == "stats":
            boson_stats[result["boson_id"]] = result
        else:
            results.append(result)
    
    elapsed = time.time() - start_time
    
    # Wait for all processes
    for p in processes:
        p.join()
    
    print(f"\n✅ Processed {len(results)} objects in {elapsed:.2f}s")
    print(f"   Throughput: {len(results)/elapsed:.0f} objects/sec")
    
    # Statistics
    print("\n" + "=" * 70)
    print("📊 Boson Statistics:")
    
    for boson_id in sorted(boson_stats.keys()):
        stats = boson_stats[boson_id]
        print(f"  Boson {boson_id:2d}: Center {stats['center'][:3]}... | "
              f"Processed {stats['objects_processed']:4d} objects")
    
    # Analyze distribution
    print("\n📈 Object Distribution by Boson:")
    
    by_boson = {}
    in_radius = {}
    
    for result in results:
        bid = result["boson_id"]
        by_boson[bid] = by_boson.get(bid, 0) + 1
        if result["in_radius"]:
            in_radius[bid] = in_radius.get(bid, 0) + 1
    
    for boson_id in sorted(by_boson.keys())[:10]:
        count = by_boson[boson_id]
        in_r = in_radius.get(boson_id, 0)
        print(f"  Boson {boson_id:2d}: {count:4d} objects ({in_r:3d} in radius)")
    
    # Save results
    output = {
        "num_bosons": num_bosons,
        "num_objects": len(results),
        "elapsed_seconds": elapsed,
        "throughput": len(results) / elapsed,
        "boson_stats": boson_stats,
        "sample_results": results[:20]
    }
    
    Path("boson_processing.json").write_text(json.dumps(output, indent=2, default=str))
    
    print(f"\n💾 Results saved to boson_processing.json")
    
    print("\n🔵 Boson Properties:")
    print("  - Self-centered: Each has own center in 71^7 space")
    print("  - Parallel: All 24 run simultaneously")
    print("  - Independent: Each processes from its perspective")
    print("  - Radius: Explores within distance from center")
    
    print("\n💡 Scaling to 8M Objects:")
    print(f"  - Current: {len(results)} objects in {elapsed:.2f}s")
    print(f"  - Estimated 8M: {8_000_000 / (len(results)/elapsed):.0f}s = "
          f"{8_000_000 / (len(results)/elapsed) / 60:.1f} minutes")
    
    print("\n∞ 24 Bosons. Self-Centered. Parallel. 71^7 Space. ∞")

if __name__ == "__main__":
    main()
