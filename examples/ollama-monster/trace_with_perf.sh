#!/usr/bin/env bash
set -e

PROMPT="$1"
OUTPUT_DIR="perf_traces"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%s)
PERF_DATA="$OUTPUT_DIR/perf_${TIMESTAMP}.data"
PERF_SCRIPT="$OUTPUT_DIR/perf_${TIMESTAMP}.script"
RESPONSE_FILE="$OUTPUT_DIR/response_${TIMESTAMP}.json"

echo "🔬 Tracing model execution with perf"
echo "Prompt: $PROMPT"
echo ""

# Start perf recording in background
sudo perf record -e cycles,instructions,cache-misses,cache-references \
    -g -o "$PERF_DATA" -a &
PERF_PID=$!

sleep 1

# Run the model query
curl -s http://localhost:11434/api/generate -d "{
  \"model\": \"qwen2.5:3b\",
  \"prompt\": \"$PROMPT\",
  \"stream\": false
}" > "$RESPONSE_FILE"

# Stop perf
sleep 1
sudo kill -INT $PERF_PID
wait $PERF_PID 2>/dev/null || true

echo "✓ Perf data captured: $PERF_DATA"

# Generate perf script
sudo perf script -i "$PERF_DATA" > "$PERF_SCRIPT"
echo "✓ Perf script: $PERF_SCRIPT"

# Extract response
cat "$RESPONSE_FILE" | jq -r '.response'

echo ""
echo "📊 Analysis:"
echo "  Perf data: $PERF_DATA"
echo "  Script: $PERF_SCRIPT"
echo "  Response: $RESPONSE_FILE"
