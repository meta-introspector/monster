#!/bin/bash
# Vile code containment wrapper script

set -euo pipefail

VILE_CODE="$1"
THREAT_LEVEL="${2:-unknown}"

echo "=== Vile Code Containment ==="
echo "Code: $VILE_CODE"
echo "Threat Level: $THREAT_LEVEL"
echo "Time: $(date -Iseconds)"
echo

# Create containment directory
CONTAIN_DIR="/tmp/vile_containment_$$"
mkdir -p "$CONTAIN_DIR"
cd "$CONTAIN_DIR"

# Copy vile code (read-only)
cp "$VILE_CODE" ./vile_code.bin
chmod 500 ./vile_code.bin

# Apply SELinux label
chcon -t vile_code_exec_t ./vile_code.bin 2>/dev/null || true

# Create isolated namespace
unshare --user --pid --net --mount --uts --ipc --map-root-user --fork bash -c "
    # Mount read-only root
    mount --bind / /mnt 2>/dev/null || true
    mount -o remount,ro /mnt 2>/dev/null || true
    
    # Minimal /tmp
    mount -t tmpfs -o size=1M,noexec,nodev,nosuid tmpfs /tmp 2>/dev/null || true
    
    # Drop capabilities
    capsh --drop=all -- -c '
        # Run with timeout and seccomp
        timeout 10 ./vile_seccomp ./vile_code.bin 2>&1 | tee output.log
    ' || echo 'Contained execution completed'
"

# Cleanup
echo
echo "=== Cleanup ==="
shred -vfz -n 3 ./vile_code.bin 2>/dev/null || rm -f ./vile_code.bin
cd /
rm -rf "$CONTAIN_DIR"

echo "Containment complete: $(date -Iseconds)"
