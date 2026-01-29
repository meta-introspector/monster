#!/usr/bin/env bash

echo "🔬 SCANNING ALL SOURCES FOR MONSTER PRIMES"
echo "==========================================="
echo ""

SOURCES=("spectral" "vericoding" "tex_lean_retriever" "FormalBook" "ProofNet")

for source in "${SOURCES[@]}"; do
    if [ -d "$source" ]; then
        echo "📂 $source"
        echo "---"
        
        # Count files
        lean_count=$(find "$source" -name "*.lean" 2>/dev/null | wc -l)
        hlean_count=$(find "$source" -name "*.hlean" 2>/dev/null | wc -l)
        py_count=$(find "$source" -name "*.py" 2>/dev/null | wc -l)
        
        echo "  .lean files: $lean_count"
        echo "  .hlean files: $hlean_count"
        echo "  .py files: $py_count"
        
        # Scan for Monster primes
        total_files=$((lean_count + hlean_count + py_count))
        if [ $total_files -gt 0 ]; then
            echo "  Monster primes:"
            for prime in 2 3 5 7 11 13 17 19 23 29 31 41 47 59 71; do
                count=$(grep -r "\b${prime}\b" "$source" --include="*.lean" --include="*.hlean" --include="*.py" 2>/dev/null | wc -l)
                if [ $count -gt 0 ]; then
                    printf "    %3d: %5d mentions\n" $prime $count
                fi
            done
            
            # Check for 71 specifically
            has_71=$(grep -r "\b71\b" "$source" --include="*.lean" --include="*.hlean" --include="*.py" 2>/dev/null | wc -l)
            if [ $has_71 -gt 0 ]; then
                echo "  👹 MONSTER PRIME 71 FOUND!"
            fi
        fi
        echo ""
    fi
done

echo "✅ Scan complete!"
