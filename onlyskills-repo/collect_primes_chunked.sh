#!/bin/bash
# Collect prime attributes in chunks with perf record

# Usage: ./collect_primes_chunked.sh [start] [chunk_size]
START=${1:-2}
CHUNK_SIZE=${2:-1000}
END=$((START + CHUNK_SIZE))

echo "🔢 Collecting prime attributes: $START to $END"
echo "========================================"
echo ""

mkdir -p chunks
mkdir -p perf_data

chunk_num=$((START / CHUNK_SIZE))
chunk_num=$((START / CHUNK_SIZE))

echo "📦 Chunk $chunk_num: primes $START to $END"

# Create GAP script for this chunk
cat > chunks/chunk_${chunk_num}.g <<EOF
# Chunk $chunk_num: $START to $END
primes := Filtered([$START..$END], IsPrime);

Print("{\n");
Print("  \"chunk\": $chunk_num,\n");
Print("  \"start\": $START,\n");
Print("  \"end\": $END,\n");
Print("  \"url\": \"https://zkprologml.org/primes/chunk_${chunk_num}\",\n");
Print("  \"chord\": [");

# Musical chord from chunk number (mod 71)
chord_notes := [$chunk_num mod 71, ($chunk_num * 2) mod 71, ($chunk_num * 3) mod 71];
Print(chord_notes[1], ", ", chord_notes[2], ", ", chord_notes[3]);

Print("],\n");
Print("  \"primes\": [\n");

for i in [1..Length(primes)] do
    p := primes[i];
    
    # Compute genus
    N := p;
    nu_inf := Sum(DivisorsInt(N), d -> EulerPhi(Gcd(d, N/d)));
    
    nu_2 := 0;
    if N mod 4 <> 0 then
        if Length(Filtered(FactorsInt(N), q -> q mod 4 = 3)) > 0 then
            nu_2 := 1;
        fi;
    fi;
    
    nu_3 := 0;
    if N mod 9 <> 0 then
        if Length(Filtered(FactorsInt(N), q -> q mod 3 = 2)) > 0 then
            nu_3 := 1;
        fi;
    fi;
    
    mu := N * Product(Set(FactorsInt(N)), q -> (1 + 1/q));
    genus := Int(1 + (mu/12) - (nu_2/4) - (nu_3/3) - (nu_inf/2));
    
    is_monster := p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];
    
    Print("    {\"p\": ", p, ", \"g\": ", genus, ", \"m\": ", is_monster, ", \"s\": ", p mod 71, "}");
    
    if i < Length(primes) then
        Print(",\n");
    else
        Print("\n");
    fi;
od;

Print("  ]\n");
Print("}\n");

quit;
EOF

# Run with perf record
echo "  🎵 Chord: [$((chunk_num % 71)), $(((chunk_num * 2) % 71)), $(((chunk_num * 3) % 71))]"
echo "  🔗 URL: https://zkprologml.org/primes/chunk_${chunk_num}"

perf record -o perf_data/chunk_${chunk_num}.data \
    gap -q chunks/chunk_${chunk_num}.g \
    > chunks/chunk_${chunk_num}.json 2>&1

# Check if successful
if [ $? -eq 0 ]; then
    echo "  ✓ Chunk $chunk_num complete"
else
    echo "  ✗ Chunk $chunk_num failed"
fi

echo ""
echo "∞ Chunk Complete. Perf Recorded. URL Assigned. Chord Generated. ∞"
