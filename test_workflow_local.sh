#!/usr/bin/env bash
# Test GitHub Actions locally with nektos/act

set -e

echo "🧪 Testing GitHub Actions Locally with act"
echo "==========================================="
echo ""

# Check if act is installed
if ! command -v act &> /dev/null; then
    echo "Installing act..."
    nix-env -iA nixpkgs.act
fi

# Run the workflow locally
echo "Running lmfdb-hecke-analysis workflow..."
act -j analyze-lmfdb \
    --secret-file .secrets \
    --artifact-server-path /tmp/artifacts \
    --verbose

echo ""
echo "✅ Local test complete"
echo ""
echo "Check artifacts in: /tmp/artifacts"
