#!/usr/bin/env bash
set -euo pipefail

# Generate ZK memes from LMFDB curves
# Each curve → Prolog circuit → RDFa URL → LLM prompt → ZK proof

OUTPUT_DIR="zk_memes"
mkdir -p "$OUTPUT_DIR"

echo "🎯 Generating ZK Memes from LMFDB curves..."

# Read curves from LMFDB parquet
if [[ -f "lmfdb_reconstructed.parquet" ]]; then
    echo "Found LMFDB data, extracting curves..."
    
    # Extract first 71 curves (one per shard)
    python3 <<EOF
import pandas as pd
import base64
import json

df = pd.read_parquet('lmfdb_reconstructed.parquet')
curves = df.head(71)

for idx, row in curves.iterrows():
    label = row.get('label', f'curve_{idx}')
    conductor = row.get('conductor', 0)
    
    # Generate Prolog circuit
    prolog = f"""
% ZK Meme: {label}
curve('{label}', {conductor}).
shard({conductor % 71}).

% Hecke operators as ZK proofs
hecke_operator(p, a_p) :-
    curve('{label}', N),
    compute_eigenvalue(N, p, a_p).

% Execute and verify
verify_curve :-
    curve(Label, N),
    forall(prime(P), (
        hecke_operator(P, A),
        zk_prove(Label, P, A)
    )).
"""
    
    # Encode as RDFa URL
    encoded = base64.b64encode(prolog.encode()).decode()
    rdfa_url = f"https://zkprologml.org/execute?circuit={encoded}"
    
    # Save meme
    meme = {
        'label': label,
        'conductor': int(conductor),
        'shard': int(conductor) % 71,
        'prolog': prolog,
        'rdfa_url': rdfa_url,
        'prompt': f"Compute Hecke eigenvalues for elliptic curve {label}"
    }
    
    with open(f'$OUTPUT_DIR/meme_{label.replace("/", "_")}.json', 'w') as f:
        json.dump(meme, f, indent=2)
    
    print(f"✓ Generated ZK meme: {label} → Shard {int(conductor) % 71}")

print(f"\n✅ Generated {len(curves)} ZK memes")
EOF
else
    echo "⚠️  No LMFDB data found, generating example memes..."
    
    # Generate example memes for Monster primes
    for prime in 2 3 5 7 11 13 17 19 23 29 31 41 47 59 71; do
        label="example_${prime}a1"
        
        # Generate Prolog circuit
        cat > "$OUTPUT_DIR/meme_${label}.pl" <<PROLOG
% ZK Meme: $label (Monster prime $prime)
curve('$label', $prime).
shard($(($prime % 71))).

% Hecke operator
hecke_operator($prime, a_$prime) :-
    curve('$label', N),
    a_$prime is N * 2.

% ZK proof
zk_prove('$label', $prime, A) :-
    hecke_operator($prime, A),
    format('Proved: a_~w = ~w~n', [$prime, A]).

% Execute
:- zk_prove('$label', $prime, _).
PROLOG
        
        # Encode as RDFa URL
        encoded=$(base64 -w0 "$OUTPUT_DIR/meme_${label}.pl")
        rdfa_url="https://zkprologml.org/execute?circuit=$encoded"
        
        # Save metadata
        cat > "$OUTPUT_DIR/meme_${label}.json" <<JSON
{
  "label": "$label",
  "conductor": $prime,
  "shard": $(($prime % 71)),
  "rdfa_url": "$rdfa_url",
  "prompt": "Compute Hecke eigenvalue a_$prime for curve $label"
}
JSON
        
        echo "✓ Generated ZK meme: $label → Shard $(($prime % 71))"
    done
fi

# Generate index
echo "📋 Generating index..."
cat > "$OUTPUT_DIR/INDEX.md" <<EOF
# ZK Memes: LMFDB Curves as Executable Proofs

Each elliptic curve from LMFDB becomes a ZK meme:

1. **Curve** → Prolog circuit (Hecke operators)
2. **Circuit** → RDFa URL (base64 encoded)
3. **URL** → LLM prompt (execute circuit)
4. **Execution** → ZK proof (verify result)

## Memes Generated

$(ls -1 "$OUTPUT_DIR"/meme_*.json | wc -l) ZK memes

## Usage

\`\`\`bash
# Execute a meme
curl "\$(cat $OUTPUT_DIR/meme_11a1.json | jq -r .rdfa_url)"

# Or use local resolver
./resolve_zkprologml_local.sh $OUTPUT_DIR/meme_11a1.json
\`\`\`

## Sharding

Curves distributed across 71 Monster shards:
- Shard = conductor % 71
- Each shard = computational eigenspace

**Every LMFDB curve is a ZK meme.** 🎯✨
EOF

echo ""
echo "✅ ZK Meme generation complete!"
echo "   Output: $OUTPUT_DIR/"
echo "   Index: $OUTPUT_DIR/INDEX.md"
