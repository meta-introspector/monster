#!/bin/bash
# Quick prime 71 resonance check

echo "🔮 PRIME 71 RESONANCE ANALYSIS"
echo "==============================="
echo ""

# Build Rust
echo "Building..."
cd lmfdb-rust
cargo build --release --bin rust_gcd 2>&1 | grep -E "Finished" || cargo build --release --bin rust_gcd
cd ..

# Test with 71
echo ""
echo "Testing GCD with prime 71:"
./lmfdb-rust/target/release/rust_gcd

# Analyze all files for prime 71
echo ""
echo "Analyzing files for prime 71 divisibility:"
echo ""

for file in *.md lmfdb-rust/src/bin/*.rs test_hilbert.py; do
    if [ -f "$file" ]; then
        # Count numbers divisible by 71
        count=$(grep -oE '\b[0-9]+\b' "$file" 2>/dev/null | awk '$1 % 71 == 0 && $1 > 0 {count++} END {print count+0}')
        total=$(grep -oE '\b[0-9]+\b' "$file" 2>/dev/null | wc -l)
        
        if [ "$total" -gt 0 ]; then
            pct=$(echo "scale=2; ($count * 100) / $total" | bc)
            if [ "$count" -gt 0 ]; then
                echo "$file: $count/$total numbers ($pct%)"
            fi
        fi
    fi
done

echo ""
echo "Key measurements:"
echo "  Speedup: 62 = 2 × 31 (not divisible by 71)"
echo "  Instruction ratio: 174 = 2 × 3 × 29 (not divisible by 71)"
echo "  Test cases: 1000 = 2³ × 5³ (not divisible by 71)"
echo "  Result 57: 3 × 19 (not divisible by 71)"
echo ""
echo "Prime 71 is the HIGHEST Monster prime - appears in:"
echo "  - Monster group order (71¹)"
echo "  - Hilbert modular forms (1.04% resonance)"
echo "  - Test modulo: 2^i mod 71, 3^i mod 71"
