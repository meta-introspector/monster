#!/usr/bin/env bash
# Experiment 02: Count Conjugacy Classes

set -e

EXPERIMENT="02_conjugacy_classes"
OUTPUT_DIR="${EXPERIMENT_DIR:-./experiments}/results"
mkdir -p "$OUTPUT_DIR"

echo "🧪 Experiment 02: Conjugacy Classes"
echo "===================================="
echo ""

cat > /tmp/conjugacy.g << 'GAP'
LoadPackage("ctbllib");
Print("Loading Monster character table...\n");

ct := CharacterTable("M");
if ct = fail then
    Print("ERROR: Failed to load character table\n");
    QUIT_GAP(1);
fi;

num_classes := NrConjugacyClasses(ct);
Print("Conjugacy classes: ", num_classes, "\n");

expected := 194;
Print("Expected:          ", expected, "\n");

if num_classes = expected then
    Print("✓ VERIFIED: 194 conjugacy classes!\n");
    result := "PASS";
else
    Print("✗ FAILED: Mismatch!\n");
    result := "FAIL";
fi;

# Get class sizes
sizes := SizesConjugacyClasses(ct);
Print("Sample class sizes: ", sizes{[1..5]}, "\n");

json := Concatenation(
    "{\n",
    "  \"experiment\": \"02_conjugacy_classes\",\n",
    "  \"result\": \"", result, "\",\n",
    "  \"num_classes\": ", String(num_classes), ",\n",
    "  \"expected\": ", String(expected), ",\n",
    "  \"match\": ", String(num_classes = expected), "\n",
    "}\n"
);

PrintTo("$OUTPUT_DIR/$EXPERIMENT.json", json);
Print("\n✓ Results saved\n");

QUIT_GAP(0);
GAP

echo "Running GAP..."
gap -q /tmp/conjugacy.g

echo ""
echo "Converting to Parquet..."
python3 << 'PYTHON'
import json
import pandas as pd
from datetime import datetime

with open("$OUTPUT_DIR/$EXPERIMENT.json") as f:
    data = json.load(f)

df = pd.DataFrame([{
    'experiment': data['experiment'],
    'result': data['result'],
    'num_classes': data['num_classes'],
    'expected': data['expected'],
    'match': data['match'],
    'timestamp': datetime.now().isoformat()
}])

df.to_parquet("$OUTPUT_DIR/$EXPERIMENT.parquet", index=False)
print(f"✓ Saved to $OUTPUT_DIR/$EXPERIMENT.parquet")
PYTHON

echo ""
echo "✅ Experiment complete!"
