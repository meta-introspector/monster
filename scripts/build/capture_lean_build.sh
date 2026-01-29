#!/usr/bin/env bash
# Capture telemetry for Lean4 builds

set -e

START=$(date +%s%3N)

echo "📊 Capturing Lean4 build telemetry..."

# Run lake build
cd MonsterLean
lake build 2>&1 | tee ../lean_build.log
BUILD_SUCCESS=${PIPESTATUS[0]}
cd ..

END=$(date +%s%3N)
DURATION=$((END - START))

# Extract proof statistics
python3 << EOF
import pandas as pd
from datetime import datetime
import subprocess
import os
import json
import re
from pathlib import Path

# Get git info
try:
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
except:
    git_commit = "unknown"
    git_branch = "unknown"

# Parse build log
with open('lean_build.log') as f:
    log_content = f.read()

# Count proofs
theorems = len(re.findall(r'\btheorem\b', log_content))
axioms = len(re.findall(r'\baxiom\b', log_content))
lemmas = len(re.findall(r'\blemma\b', log_content))

# Count Lean files
lean_files = list(Path('MonsterLean').rglob('*.lean'))
total_lines = sum(len(open(f).readlines()) for f in lean_files)

# Get runner info
runner = os.environ.get('GITHUB_ACTIONS', 'false')
if runner == 'true':
    runner_type = 'github-hosted'
    runner_name = os.environ.get('RUNNER_NAME', 'unknown')
else:
    runner_type = 'local'
    runner_name = os.uname().nodename

# Create telemetry record
record = {
    'timestamp': datetime.now().isoformat(),
    'build_type': 'lean',
    'duration_ms': $DURATION,
    'success': $BUILD_SUCCESS == 0,
    'git_commit': git_commit,
    'git_branch': git_branch,
    'runner_type': runner_type,
    'runner_name': runner_name,
    'theorem_count': theorems,
    'axiom_count': axioms,
    'lemma_count': lemmas,
    'file_count': len(lean_files),
    'total_lines': total_lines,
    'log_file': 'lean_build.log'
}

# Save to parquet
df = pd.DataFrame([record])
df.to_parquet('lean_build_telemetry.parquet', index=False)

print(f"✅ Telemetry saved: {theorems} theorems, {axioms} axioms, {lemmas} lemmas, {$DURATION}ms")
EOF

exit $BUILD_SUCCESS
