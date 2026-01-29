#!/usr/bin/env bash
# Capture telemetry for Rust builds

set -e

BUILD_TYPE=${1:-"release"}
START=$(date +%s%3N)

echo "📊 Capturing Rust build telemetry..."

# Run build
if [ "$BUILD_TYPE" = "release" ]; then
    cargo build --release 2>&1 | tee rust_build.log
    BUILD_SUCCESS=${PIPESTATUS[0]}
else
    cargo build 2>&1 | tee rust_build.log
    BUILD_SUCCESS=${PIPESTATUS[0]}
fi

END=$(date +%s%3N)
DURATION=$((END - START))

# Extract build info
python3 << EOF
import pandas as pd
from datetime import datetime
import subprocess
import os
import json

# Get git info
try:
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
except:
    git_commit = "unknown"
    git_branch = "unknown"

# Get nix info
try:
    nix_hash = subprocess.check_output([
        'nix', 'eval', '--raw', '.#devShells.x86_64-linux.default.drvPath'
    ]).decode().strip()
except:
    nix_hash = "unknown"

# Parse build log for artifacts
artifacts = []
with open('rust_build.log') as f:
    for line in f:
        if 'Finished' in line and 'target' in line:
            artifacts.append(line.strip())

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
    'build_type': 'rust',
    'build_mode': '$BUILD_TYPE',
    'duration_ms': $DURATION,
    'success': $BUILD_SUCCESS == 0,
    'git_commit': git_commit,
    'git_branch': git_branch,
    'runner_type': runner_type,
    'runner_name': runner_name,
    'nix_hash': nix_hash,
    'artifacts': json.dumps(artifacts),
    'log_file': 'rust_build.log'
}

# Save to parquet
df = pd.DataFrame([record])
df.to_parquet('rust_build_telemetry.parquet', index=False)

print(f"✅ Telemetry saved: {len(artifacts)} artifacts, {$DURATION}ms")
EOF

exit $BUILD_SUCCESS
