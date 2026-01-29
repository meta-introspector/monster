#!/usr/bin/env bash
set -e

PROMPT="$1"
OUTPUT_DIR="perf_traces"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%s)
PERF_DATA="$OUTPUT_DIR/perf_regs_${TIMESTAMP}.data"
PERF_SCRIPT="$OUTPUT_DIR/perf_regs_${TIMESTAMP}.script"
RESPONSE_FILE="$OUTPUT_DIR/response_${TIMESTAMP}.json"
PROMPT_FILE="$OUTPUT_DIR/prompt_${TIMESTAMP}.txt"

echo "🔬 Tracing model execution with perf (registers + stack)"
echo "Prompt: $PROMPT"
echo ""

# Save prompt
echo "$PROMPT" > "$PROMPT_FILE"

# Start perf recording with register capture
# Focus on cycles (hottest paths = matrix multiplications)
sudo perf record \
    -e cycles:u \
    -c 10000 \
    --intr-regs=AX,BX,CX,DX,SI,DI,R8,R9,R10,R11,R12,R13,R14,R15 \
    -o "$PERF_DATA" \
    -a &
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

# Generate perf script with registers
sudo perf script -i "$PERF_DATA" -F ip,sym,period,iregs > "$PERF_SCRIPT"
echo "✓ Perf script with registers: $PERF_SCRIPT"

# Extract response
cat "$RESPONSE_FILE" | jq -r '.response'

echo ""
echo "📊 Analysis:"
wc -l "$PERF_SCRIPT"
echo "  Prompt: $PROMPT_FILE"
echo "  Perf data: $PERF_DATA"
echo "  Script: $PERF_SCRIPT"
echo "  Response: $RESPONSE_FILE"
