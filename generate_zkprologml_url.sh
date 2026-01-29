#!/usr/bin/env bash
# Generate zkprologml RDFa URL for LLM call

set -e

# Input arguments
MODEL="${1:-gpt-4}"
PROMPT_FILE="${2:-datasets/prompts/text_prompt.txt}"
CONTEXT_FILE="${3:-datasets/audio/monster_walk_metadata.json}"
PROOF_FILE="${4:-MonsterLean/MonsterHarmonics.lean}"
TIMESTAMP="${5:-$(date +%Y%m%d_%H%M%S)}"

echo "🔗 Generating zkprologml RDFa URL"
echo "=================================="
echo "Model: $MODEL"
echo "Prompt: $PROMPT_FILE"
echo "Context: $CONTEXT_FILE"
echo "Proof: $PROOF_FILE"
echo ""

# Read and encode prompt
if [ -f "$PROMPT_FILE" ]; then
    PROMPT=$(cat "$PROMPT_FILE")
    PROMPT_B64=$(echo "$PROMPT" | base64 -w0)
else
    echo "⚠️  Prompt file not found"
    exit 1
fi

# Read and encode context
if [ -f "$CONTEXT_FILE" ]; then
    CONTEXT=$(cat "$CONTEXT_FILE")
    CONTEXT_B64=$(echo "$CONTEXT" | base64 -w0)
else
    echo "⚠️  Context file not found"
    CONTEXT_B64=""
fi

# Generate proof hash
if [ -f "$PROOF_FILE" ]; then
    PROOF_HASH=$(sha256sum "$PROOF_FILE" | cut -d' ' -f1)
else
    echo "⚠️  Proof file not found"
    PROOF_HASH="none"
fi

# Generate zkprologml URL
ZKPROLOGML_URL="zkprologml://llm.call/text-generation?model=$MODEL&prompt=$PROMPT_B64&context=$CONTEXT_B64&proof=sha256:$PROOF_HASH&verify=zkproof&timestamp=$TIMESTAMP"

# Output directory
OUTPUT_DIR="datasets/llm_calls"
mkdir -p "$OUTPUT_DIR"

# Store URL
echo "$ZKPROLOGML_URL" > "$OUTPUT_DIR/call_$TIMESTAMP.zkprologml"

# Generate RDFa
cat > "$OUTPUT_DIR/call_$TIMESTAMP.rdfa" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:zkp="http://zkprologml.org/ns#">
<head>
  <title>zkprologml LLM Call - $TIMESTAMP</title>
</head>
<body vocab="http://zkprologml.org/ns#">
  
  <div typeof="LLMCall" resource="$ZKPROLOGML_URL">
    <h1>LLM Call: Text Generation</h1>
    
    <dl>
      <dt>Model</dt>
      <dd property="model">$MODEL</dd>
      
      <dt>Prompt</dt>
      <dd property="prompt" content="$PROMPT_B64">
        <pre>$PROMPT</pre>
      </dd>
      
      <dt>Context</dt>
      <dd property="context" content="$CONTEXT_B64">
        <pre>$(echo "$CONTEXT" | head -10)...</pre>
      </dd>
      
      <dt>Proof</dt>
      <dd property="proof" content="sha256:$PROOF_HASH">
        <code>$PROOF_FILE</code>
        <br/>Hash: <code>$PROOF_HASH</code>
      </dd>
      
      <dt>Verification Method</dt>
      <dd property="verify">zkproof</dd>
      
      <dt>Timestamp</dt>
      <dd property="timestamp">$TIMESTAMP</dd>
      
      <dt>Status</dt>
      <dd property="status">suspended</dd>
    </dl>
    
    <h2>Resolution</h2>
    <p>This call is suspended. Resolve via:</p>
    <ul>
      <li>GitHub: Push to trigger CI/CD workflow</li>
      <li>Local: Run <code>./resolve_zkprologml_local.sh $OUTPUT_DIR/call_$TIMESTAMP.zkprologml</code></li>
    </ul>
    
  </div>
  
</body>
</html>
EOF

# Generate JSON metadata
cat > "$OUTPUT_DIR/call_$TIMESTAMP.json" << EOF
{
  "url": "$ZKPROLOGML_URL",
  "model": "$MODEL",
  "prompt": "$PROMPT",
  "context_file": "$CONTEXT_FILE",
  "proof_file": "$PROOF_FILE",
  "proof_hash": "$PROOF_HASH",
  "verify": "zkproof",
  "timestamp": "$TIMESTAMP",
  "status": "suspended"
}
EOF

echo "✓ zkprologml URL: $OUTPUT_DIR/call_$TIMESTAMP.zkprologml"
echo "✓ RDFa: $OUTPUT_DIR/call_$TIMESTAMP.rdfa"
echo "✓ JSON: $OUTPUT_DIR/call_$TIMESTAMP.json"
echo ""
echo "URL:"
echo "$ZKPROLOGML_URL"
echo ""
echo "⏸️  Execution suspended"
echo ""
echo "To resolve:"
echo "  GitHub: git push (triggers workflow)"
echo "  Local:  ./resolve_zkprologml_local.sh $OUTPUT_DIR/call_$TIMESTAMP.zkprologml"
