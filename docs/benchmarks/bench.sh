#!/bin/bash
# vMCU vs standard performance benchmark on QEMU.
# Usage: bench.sh <report-name> [machine] [cpu]
set -euo pipefail

NAME="$1"
MACHINE="${2:-mps2-an386}"
CPU="${3:-cortex-m4}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
EXAMPLE="$REPO/examples/qemu-vmcu-auto"
REPORT_DIR="$REPO/docs/benchmarks"
REPORT="$REPORT_DIR/$NAME.txt"
KERNEL="$EXAMPLE/target/thumbv7em-none-eabi/release/qemu-vmcu-auto"

mkdir -p "$REPORT_DIR"

{
echo "# vMCU performance report: $NAME"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Machine: $MACHINE  CPU: $CPU"
echo

run_variant() {
    local label="$1" feature="$2"
    echo "## $label"
    cd "$EXAMPLE"
    if [ -n "$feature" ]; then
        BUILD="$(cargo build --release --target thumbv7em-none-eabi --features "$feature" 2>&1)"
    else
        BUILD="$(cargo build --release --target thumbv7em-none-eabi 2>&1)"
    fi
    echo "$BUILD" | grep -E "Flash Usage|RAM Usage|vMCU" || true
    RUN="$(timeout 300s qemu-system-arm -machine "$MACHINE" -cpu "$CPU" -nographic \
        -semihosting-config enable=on,target=native -kernel "$KERNEL" 2>&1 || true)"
    echo "$RUN" | grep -E "^[a-z].*:|latency|output|PASSED|FAILED" || true
    echo
}

run_variant "vMCU auto" ""
run_variant "Standard" "standard"
} > "$REPORT"
echo "report saved: $REPORT"