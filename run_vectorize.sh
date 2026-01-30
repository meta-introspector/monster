#!/bin/bash
# Run vectorization in background

cd /home/mdupont/experiments/monster

echo "🔥 Starting vectorization: $(date)"

# Build and run
cargo build --release --bin vectorize_all_parquets 2>&1 | tail -5

if [ -f target/release/vectorize_all_parquets ]; then
    ./target/release/vectorize_all_parquets &
    PID=$!
    echo "✅ Running with PID: $PID"
    echo $PID > vectorize.pid
else
    echo "❌ Build failed"
fi
