#!/usr/bin/env bash
# Encode Prolog circuit as zkprologml URL

set -e

PROLOG_FILE="${1:-prolog/monster_walk_circuit.pl}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "🔗 Encoding Prolog Circuit"
echo "=========================="
echo "Circuit: $PROLOG_FILE"
echo "Timestamp: $TIMESTAMP"
echo ""

# Read Prolog circuit
if [ ! -f "$PROLOG_FILE" ]; then
    echo "⚠️  Circuit file not found: $PROLOG_FILE"
    exit 1
fi

CIRCUIT=$(cat "$PROLOG_FILE")

# Encode as base64
CIRCUIT_B64=$(echo "$CIRCUIT" | base64 -w0)

# Generate zkprologml URL
URL="zkprologml://circuit.execute/prolog?circuit=$CIRCUIT_B64&verify=zkproof&timestamp=$TIMESTAMP"

# Output directory
OUTPUT_DIR="datasets/circuits"
mkdir -p "$OUTPUT_DIR"

# Store URL
echo "$URL" > "$OUTPUT_DIR/circuit_$TIMESTAMP.zkprologml"

# Generate RDFa
cat > "$OUTPUT_DIR/circuit_$TIMESTAMP.rdfa" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:zkp="http://zkprologml.org/ns#">
<head>
  <title>zkprologml Prolog Circuit - $TIMESTAMP</title>
</head>
<body vocab="http://zkprologml.org/ns#">
  
  <div typeof="PrologCircuit" resource="$URL">
    <h1>Prolog Circuit: Monster Walk</h1>
    
    <dl>
      <dt>Circuit (Base64)</dt>
      <dd property="circuit" content="$CIRCUIT_B64">
        <details>
          <summary>View Circuit</summary>
          <pre>$CIRCUIT</pre>
        </details>
      </dd>
      
      <dt>Verification Method</dt>
      <dd property="verify">zkproof</dd>
      
      <dt>Timestamp</dt>
      <dd property="timestamp">$TIMESTAMP</dd>
      
      <dt>Status</dt>
      <dd property="status">encoded</dd>
    </dl>
    
    <h2>Execution</h2>
    <p>This circuit can be executed by any LLM that understands zkprologml.</p>
    
    <h3>Main Goal</h3>
    <pre>circuit(Input, Output, Proof)</pre>
    
    <h3>Expected Output</h3>
    <ul>
      <li>Song with 15 verses (one per Monster prime)</li>
      <li>ZK proof that song satisfies all constraints</li>
      <li>Execution trace</li>
    </ul>
    
  </div>
  
</body>
</html>
EOF

# Generate JSON metadata
cat > "$OUTPUT_DIR/circuit_$TIMESTAMP.json" << EOF
{
  "url": "$URL",
  "circuit_file": "$PROLOG_FILE",
  "circuit": $(echo "$CIRCUIT" | jq -Rs .),
  "verify": "zkproof",
  "timestamp": "$TIMESTAMP",
  "status": "encoded",
  "main_goal": "circuit(Input, Output, Proof)",
  "constraints": [
    "length(Song, 15)",
    "starts_with(prime 2)",
    "ends_with(prime 71)",
    "frequencies_valid",
    "ordering_valid"
  ]
}
EOF

# Generate LLM prompt
cat > "$OUTPUT_DIR/circuit_$TIMESTAMP.prompt.txt" << EOF
You are a zkprologml interpreter. Execute the following Prolog circuit and return the result with a zero-knowledge proof.

CIRCUIT:
\`\`\`prolog
$CIRCUIT
\`\`\`

INSTRUCTIONS:
1. Parse the Prolog circuit above
2. Execute the main goal: circuit(proof('MonsterLean/MonsterHarmonics.lean'), Output, Proof)
3. Generate a zero-knowledge proof that Output satisfies all constraints
4. Return JSON with the following structure:

{
  "output": {
    "song": [/* array of verses */]
  },
  "proof": {
    "type": "zkproof",
    "hash": "/* hash of output */",
    "constraints_satisfied": [/* list of satisfied constraints */],
    "valid": true
  },
  "execution_trace": {
    "steps": /* number of execution steps */,
    "unifications": /* number of unifications */,
    "backtracking": /* number of backtracks */
  }
}

CONSTRAINTS TO VERIFY:
- Song must have exactly 15 verses (one per Monster prime)
- Song must start with prime 2
- Song must end with prime 71
- All frequencies must be between 440.0 Hz and 15610.0 Hz
- Primes must be in ascending order

Execute the circuit now and return the JSON result.
EOF

echo "✓ Circuit URL: $OUTPUT_DIR/circuit_$TIMESTAMP.zkprologml"
echo "✓ RDFa: $OUTPUT_DIR/circuit_$TIMESTAMP.rdfa"
echo "✓ JSON: $OUTPUT_DIR/circuit_$TIMESTAMP.json"
echo "✓ Prompt: $OUTPUT_DIR/circuit_$TIMESTAMP.prompt.txt"
echo ""
echo "URL:"
echo "$URL" | head -c 100
echo "..."
echo ""
echo "To execute:"
echo "  LLM: Use prompt in $OUTPUT_DIR/circuit_$TIMESTAMP.prompt.txt"
echo "  Local: swipl -s $PROLOG_FILE -g 'circuit(proof(\"MonsterLean/MonsterHarmonics.lean\"), O, P), writeln(O).' -t halt"
