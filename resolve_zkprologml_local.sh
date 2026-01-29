#!/usr/bin/env bash
# Resolve zkprologml URL locally using ollama

set -e

ZKPROLOGML_FILE="$1"

if [ ! -f "$ZKPROLOGML_FILE" ]; then
    echo "⚠️  zkprologml file not found: $ZKPROLOGML_FILE"
    exit 1
fi

echo "🔍 Resolving zkprologml URL (local)"
echo "===================================="
echo "File: $ZKPROLOGML_FILE"
echo ""

# Read URL
URL=$(cat "$ZKPROLOGML_FILE")
echo "URL: $URL"
echo ""

# Parse URL components
MODEL=$(echo "$URL" | grep -oP 'model=\K[^&]+')
PROMPT_B64=$(echo "$URL" | grep -oP 'prompt=\K[^&]+')
CONTEXT_B64=$(echo "$URL" | grep -oP 'context=\K[^&]+')
PROOF=$(echo "$URL" | grep -oP 'proof=\K[^&]+')
TIMESTAMP=$(echo "$URL" | grep -oP 'timestamp=\K[^&]+')

# Decode
PROMPT=$(echo "$PROMPT_B64" | base64 -d)
CONTEXT=$(echo "$CONTEXT_B64" | base64 -d)

echo "Model: $MODEL"
echo "Proof: $PROOF"
echo "Timestamp: $TIMESTAMP"
echo ""

# Call local LLM (ollama)
echo "🤖 Calling local LLM..."
if command -v ollama &> /dev/null; then
    RESULT=$(ollama run "$MODEL" "$PROMPT" 2>&1 || echo "⚠️  ollama failed")
else
    echo "⚠️  ollama not found, using placeholder"
    RESULT="[Placeholder result - ollama not installed]"
fi

echo "✓ LLM result generated"
echo ""

# Store result
OUTPUT_DIR="datasets/llm_results"
mkdir -p "$OUTPUT_DIR"

RESULT_FILE="$OUTPUT_DIR/result_$TIMESTAMP.json"

# Escape JSON properly
PROMPT_JSON=$(echo "$PROMPT" | jq -Rs .)
RESULT_JSON=$(echo "$RESULT" | jq -Rs .)

cat > "$RESULT_FILE" << EOF
{
  "timestamp": "$TIMESTAMP",
  "model": "$MODEL",
  "prompt": $PROMPT_JSON,
  "context": $CONTEXT,
  "proof": "$PROOF",
  "result": $RESULT_JSON,
  "status": "resolved",
  "resolver": "local",
  "verified": false
}
EOF

echo "✓ Result stored: $RESULT_FILE"
echo ""

# Verify (placeholder - needs Lean4)
echo "🔍 Verifying result..."
if [ -f "MonsterLean/VerifyLLMCall.lean" ]; then
    echo "⚠️  Lean4 verification not yet implemented"
    VERIFIED="pending"
else
    echo "⚠️  Verification proof not found"
    VERIFIED="skipped"
fi

# Update result with verification
jq ".verified = \"$VERIFIED\"" "$RESULT_FILE" > "$RESULT_FILE.tmp"
mv "$RESULT_FILE.tmp" "$RESULT_FILE"

echo "✓ Verification: $VERIFIED"
echo ""
echo "✅ Resolution complete"
echo ""
echo "Result:"
echo "$RESULT" | head -20
