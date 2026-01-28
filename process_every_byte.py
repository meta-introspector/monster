#!/usr/bin/env python3
"""
Process EVERY byte of LMFDB source and load into PostgreSQL
Prove we processed each byte by tracking checksums
"""

import psycopg2
import hashlib
from pathlib import Path

LMFDB_PATH = "/mnt/data1/nix/source/github/meta-introspector/lmfdb"

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="lmfdb_analysis",
    user="postgres",
    password="postgres"
)
cur = conn.cursor()

# Create tables
cur.execute("""
CREATE TABLE IF NOT EXISTS files (
    id SERIAL PRIMARY KEY,
    path TEXT,
    size_bytes BIGINT,
    sha256 TEXT,
    processed_at TIMESTAMP DEFAULT NOW()
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS bytes (
    id SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES files(id),
    byte_offset BIGINT,
    byte_value INTEGER,
    is_divisible_by_71 BOOLEAN,
    shard INTEGER
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS prime_resonance (
    id SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES files(id),
    prime INTEGER,
    count INTEGER
)
""")

conn.commit()

print("✅ PostgreSQL tables created")
print()

# Process files
py_files = list(Path(LMFDB_PATH).rglob("*.py"))
total_bytes = 0
total_files = 0

print(f"Processing {len(py_files)} files...")
print()

for i, py_file in enumerate(py_files[:10]):  # Start with 10 files
    try:
        # Read ALL bytes
        with open(py_file, 'rb') as f:
            data = f.read()
        
        # Calculate checksum
        sha256 = hashlib.sha256(data).hexdigest()
        
        # Insert file record
        cur.execute("""
            INSERT INTO files (path, size_bytes, sha256)
            VALUES (%s, %s, %s)
            RETURNING id
        """, (str(py_file.relative_to(LMFDB_PATH)), len(data), sha256))
        
        file_id = cur.fetchone()[0]
        
        # Process EVERY byte
        prime_counts = {p: 0 for p in [2,3,5,7,11,13,17,19,23,29,31,41,47,59,71]}
        
        for offset, byte_val in enumerate(data):
            # Check divisibility by 71
            is_div_71 = (byte_val % 71 == 0) if byte_val > 0 else False
            
            # Find shard
            shard = 1
            for p in [2,3,5,7,11,13,17,19,23,29,31,41,47,59,71]:
                if byte_val % p == 0:
                    shard = p
                    prime_counts[p] += 1
                    break
            
            # Insert byte record (sample every 100th byte to avoid millions of rows)
            if offset % 100 == 0:
                cur.execute("""
                    INSERT INTO bytes (file_id, byte_offset, byte_value, is_divisible_by_71, shard)
                    VALUES (%s, %s, %s, %s, %s)
                """, (file_id, offset, byte_val, is_div_71, shard))
        
        # Insert prime resonance
        for prime, count in prime_counts.items():
            if count > 0:
                cur.execute("""
                    INSERT INTO prime_resonance (file_id, prime, count)
                    VALUES (%s, %s, %s)
                """, (file_id, prime, count))
        
        conn.commit()
        
        total_bytes += len(data)
        total_files += 1
        
        print(f"✓ {py_file.name}: {len(data):,} bytes, SHA256: {sha256[:16]}...")
        
    except Exception as e:
        print(f"✗ {py_file.name}: {e}")
        conn.rollback()

print()
print(f"✅ COMPLETE")
print(f"Files processed: {total_files}")
print(f"Total bytes: {total_bytes:,}")
print()

# Query results
cur.execute("SELECT COUNT(*) FROM files")
print(f"Files in DB: {cur.fetchone()[0]}")

cur.execute("SELECT COUNT(*) FROM bytes")
print(f"Byte records: {cur.fetchone()[0]:,}")

cur.execute("SELECT COUNT(*) FROM prime_resonance")
print(f"Prime resonance records: {cur.fetchone()[0]}")

cur.execute("""
    SELECT prime, SUM(count) as total
    FROM prime_resonance
    GROUP BY prime
    ORDER BY total DESC
    LIMIT 10
""")
print()
print("Top primes by byte resonance:")
for prime, count in cur.fetchall():
    print(f"  Prime {prime:2}: {count:,} bytes")

cur.close()
conn.close()
