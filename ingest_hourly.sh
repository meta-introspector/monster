#!/bin/bash
# Hourly file ingestion with Monster sharding

cd /home/mdupont/experiments/monster

echo "🔥 Hourly Monster Ingestion: $(date)"
echo "======================================================================"

# Run ingestion
cargo run --release --bin ingest_all_files

# Commit results
git add all_files_monster_shards.parquet
git commit -m "Hourly ingestion: $(date +%Y-%m-%d_%H:%M)"

echo "✅ Ingestion complete"
