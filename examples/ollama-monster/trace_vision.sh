#!/usr/bin/env bash
set -e

IMAGE_PATH="$1"
OUTPUT_DIR="vision_traces"
mkdir -p "$OUTPUT_DIR"

if [ -z "$IMAGE_PATH" ]; then
    echo "Usage: $0 <image_path>"
    exit 1
fi

TIMESTAMP=$(date +%s)
PERF_DATA="$OUTPUT_DIR/vision_${TIMESTAMP}.data"
PERF_SCRIPT="$OUTPUT_DIR/vision_${TIMESTAMP}.script"
RESPONSE_FILE="$OUTPUT_DIR/vision_response_${TIMESTAMP}.json"
IMAGE_FILE="$OUTPUT_DIR/vision_image_${TIMESTAMP}.txt"

echo "🔬 Tracing Vision Model with Image"
echo "Image: $IMAGE_PATH"
echo ""

# Save image path
echo "$IMAGE_PATH" > "$IMAGE_FILE"

# Start perf recording
sudo perf record \
    -e cycles:u \
    -c 10000 \
    --intr-regs=AX,BX,CX,DX,SI,DI,R8,R9,R10,R11,R12,R13,R14,R15 \
    -o "$PERF_DATA" \
    -a &
PERF_PID=$!

sleep 1

# Query vision model (using llava via ollama)
# Encode image to base64
IMAGE_B64=$(base64 -w 0 "$IMAGE_PATH")

curl -s http://localhost:11434/api/generate -d "{
  \"model\": \"llava:7b\",
  \"prompt\": \"Describe this image in detail. What numbers, primes, or mathematical concepts do you see?\",
  \"images\": [\"$IMAGE_B64\"],
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
echo "  Image: $IMAGE_FILE"
echo "  Perf data: $PERF_DATA"
echo "  Script: $PERF_SCRIPT"
echo "  Response: $RESPONSE_FILE"
