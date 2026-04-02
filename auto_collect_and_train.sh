#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

DATA_FILE="${DATA_FILE:-data/system_metrics.csv}"
TARGET_ROWS="${TARGET_ROWS:-2000}"
HORIZON_STEPS="${HORIZON_STEPS:-5}"
TEST_SIZE="${TEST_SIZE:-0.2}"

if [[ ! -d ".venv" ]]; then
  echo "PULSE: .venv not found. Create it first."
  exit 1
fi

source .venv/bin/activate

if ! command -v stress-ng >/dev/null 2>&1; then
  echo "PULSE: stress-ng is not installed."
  echo "Install it first: sudo apt-get install -y stress-ng"
  exit 1
fi

get_rows() {
  if [[ -f "$DATA_FILE" ]]; then
    python - <<'PY'
import pandas as pd
from pathlib import Path
p = Path('data/system_metrics.csv')
if p.exists():
    try:
        print(len(pd.read_csv(p)))
    except Exception:
        print(0)
else:
    print(0)
PY
  else
    echo 0
  fi
}

run_collect() {
  local samples="$1"
  local delay="$2"
  local scenario="$3"
  python data_collection.py --samples "$samples" --delay "$delay" --scenario "$scenario"
}

safe_wait() {
  local pid="$1"
  if kill -0 "$pid" >/dev/null 2>&1; then
    wait "$pid" || true
  fi
}

run_idle_block() {
  run_collect 120 2 idle
}

run_normal_block() {
  run_collect 200 2 normal
}

run_cpu_block() {
  local duration="$1"
  local workers="$2"
  stress-ng --cpu "$workers" --cpu-method all --timeout "${duration}s" >/dev/null 2>&1 &
  local pid=$!
  run_collect 150 1 cpu_stress
  safe_wait "$pid"
}

run_ram_block() {
  local duration="$1"
  local vm_workers="$2"
  local vm_bytes="$3"
  stress-ng --vm "$vm_workers" --vm-bytes "$vm_bytes" --timeout "${duration}s" >/dev/null 2>&1 &
  local pid=$!
  run_collect 150 1 ram_stress
  safe_wait "$pid"
}

run_mixed_block() {
  local duration="$1"
  local cpu_workers="$2"
  local vm_workers="$3"
  local vm_bytes="$4"
  stress-ng --cpu "$cpu_workers" --vm "$vm_workers" --vm-bytes "$vm_bytes" --timeout "${duration}s" >/dev/null 2>&1 &
  local pid=$!
  run_collect 150 1 mixed
  safe_wait "$pid"
}

echo "PULSE: target rows = $TARGET_ROWS"
rows="$(get_rows)"
echo "PULSE: current rows = $rows"

cycle=1
while [[ "$rows" -lt "$TARGET_ROWS" ]]; do
  echo "PULSE: cycle $cycle"

  run_idle_block
  rows="$(get_rows)"
  [[ "$rows" -ge "$TARGET_ROWS" ]] && break

  run_normal_block
  rows="$(get_rows)"
  [[ "$rows" -ge "$TARGET_ROWS" ]] && break

  run_cpu_block 220 "$((RANDOM % 6 + 2))"
  rows="$(get_rows)"
  [[ "$rows" -ge "$TARGET_ROWS" ]] && break

  run_ram_block 220 "$((RANDOM % 3 + 1))" "$((RANDOM % 46 + 35))%"
  rows="$(get_rows)"
  [[ "$rows" -ge "$TARGET_ROWS" ]] && break

  run_mixed_block 220 "$((RANDOM % 4 + 2))" "$((RANDOM % 2 + 1))" "$((RANDOM % 31 + 30))%"
  rows="$(get_rows)"
  [[ "$rows" -ge "$TARGET_ROWS" ]] && break

  cycle=$((cycle + 1))
  echo "PULSE: rows so far = $rows"
done

echo "PULSE: data target reached with rows=$rows"

echo "PULSE: preprocessing forecast dataset"
python preprocess.py --label-mode forecast --horizon-steps "$HORIZON_STEPS" --window-sizes 3,5

echo "PULSE: training models"
python train.py --test-size "$TEST_SIZE" --horizon-steps "$HORIZON_STEPS"

echo "PULSE: evaluating models"
python evaluate.py --test-size "$TEST_SIZE"

echo "PULSE: complete"
