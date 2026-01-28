#!/bin/bash
# Monitor adaptive scan progress and analyze Hecke resonance

LOG="/tmp/adaptive_scan.log"
PERF="/tmp/adaptive_scan.perf.data"

echo "🔍 Monitoring Adaptive Scan + Hecke Resonance"
echo "=============================================="
echo ""

# Wait for scan to complete
while ps aux | grep -q "[a]daptive_scan"; do
    clear
    echo "🔍 Adaptive Scan Running..."
    echo "=========================="
    echo ""
    
    # Show current phase
    tail -20 "$LOG" 2>/dev/null | grep -E "Phase|Seed|Best|Final" | tail -10
    
    # Show perf data size
    if [ -f "$PERF" ]; then
        SIZE=$(du -h "$PERF" | cut -f1)
        echo ""
        echo "📊 Perf data: $SIZE"
    fi
    
    sleep 5
done

echo ""
echo "✅ Scan Complete!"
echo ""
echo "📊 Analyzing Hecke Resonance..."
python3 /home/mdupont/experiments/monster/analyze_hecke_resonance.py "$PERF" "$LOG"

echo ""
echo "📈 Final Results:"
tail -50 "$LOG" | grep -E "Best|Final|score"
