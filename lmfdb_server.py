#!/usr/bin/env python3
"""
LMFDB Server with 71-Shard Support
Serves mathematical objects sharded by Monster primes
"""

from flask import Flask, jsonify, request
import json
from pathlib import Path

app = Flask(__name__)

# Monster primes for sharding
MONSTER_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]

# Load LMFDB inventory
LMFDB_INVENTORY = Path("/home/mdupont/experiments/monster/lmfdb-inventory")

def get_shard_for_object(obj_id: str) -> int:
    """Determine which shard (0-70) an object belongs to"""
    # Hash object ID to shard
    return hash(obj_id) % 71

def get_prime_for_shard(shard_id: int) -> int:
    """Get Monster prime for shard"""
    return MONSTER_PRIMES[shard_id % 15]

@app.route('/api/shard/<int:shard_id>')
def get_shard_data(shard_id):
    """Get all LMFDB objects for a shard"""
    if shard_id < 0 or shard_id >= 71:
        return jsonify({"error": "Invalid shard ID"}), 400
    
    # Read LMFDB inventory
    objects = []
    for md_file in LMFDB_INVENTORY.glob("*.md"):
        db_name = md_file.stem.replace("db-", "")
        
        # Check if this object belongs to this shard
        if get_shard_for_object(db_name) == shard_id:
            objects.append({
                "id": db_name,
                "shard": shard_id,
                "prime": get_prime_for_shard(shard_id),
                "file": str(md_file)
            })
    
    return jsonify({
        "shard_id": shard_id,
        "prime": get_prime_for_shard(shard_id),
        "count": len(objects),
        "objects": objects
    })

@app.route('/api/shards')
def list_shards():
    """List all 71 shards"""
    shards = []
    for i in range(71):
        shards.append({
            "id": i,
            "prime": get_prime_for_shard(i),
            "url": f"/api/shard/{i}"
        })
    
    return jsonify({
        "total_shards": 71,
        "shards": shards
    })

@app.route('/api/object/<object_id>')
def get_object(object_id):
    """Get specific LMFDB object"""
    shard_id = get_shard_for_object(object_id)
    
    return jsonify({
        "id": object_id,
        "shard": shard_id,
        "prime": get_prime_for_shard(shard_id)
    })

@app.route('/health')
def health():
    """Health check"""
    return jsonify({"status": "ok", "shards": 71})

if __name__ == '__main__':
    print("🔐 LMFDB 71-Shard Server")
    print("=" * 70)
    print(f"   Shards: 71")
    print(f"   Monster primes: {MONSTER_PRIMES}")
    print(f"   Inventory: {LMFDB_INVENTORY}")
    print()
    print("🚀 Starting server on http://localhost:5000")
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=True)
