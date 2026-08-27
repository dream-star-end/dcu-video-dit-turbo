#!/usr/bin/env bash
set -eo pipefail

source /opt/dtk/env.sh
set -u

BASE=/path/to/single_rank_baseline
mkdir -p "$BASE/i2v" "$BASE/flf"

for pid_file in "$BASE"/*-server.pid "$BASE"/*-submit.pid; do
  [ -f "$pid_file" ] || continue
  pid=$(cat "$pid_file")
  kill -TERM "$pid" 2>/dev/null || true
done

for spec in "0 8192 i2v" "1 8193 flf"; do
  set -- $spec
  gpu=$1
  port=$2
  mode=$3
  setsid env \
    H3_SINGLE_GPU="$gpu" \
    H3_SINGLE_PORT="$port" \
    H3_SINGLE_STATE="$BASE/$mode" \
    python \
    launcher/run_single.py \
    >"$BASE/$mode-server.log" 2>&1 < /dev/null &
  echo $! >"$BASE/$mode-server.pid"
done

for port in 8192 8193; do
  ready=0
  for _attempt in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$port/system_stats" >/dev/null; then
      ready=1
      break
    fi
    sleep 2
  done
  if [ "$ready" -ne 1 ]; then
    echo "port $port failed to start" >&2
    exit 1
  fi
done

install -m 0644 \
  assets/noir_start_608x352.png \
  "$BASE/i2v/input/noir_start_608x352.png"
install -m 0644 \
  assets/noir_start_608x352.png \
  "$BASE/flf/input/noir_start_608x352.png"
install -m 0644 \
  assets/noir_end_608x352.png \
  "$BASE/flf/input/noir_end_608x352.png"

python - <<'PY'
import json
from pathlib import Path

src = Path("/path/to/prompts")
dst = Path("/path/to/single_rank_baseline")
for mode in ("i2v", "flf"):
    data = json.loads((src / f"{mode}_15s_20step.json").read_text())
    data["prompt"]["6"]["inputs"].pop("benchmark_nonce", None)
    data["prompt"]["14"]["inputs"]["filename_prefix"] = (
        f"single_baseline/{mode}_15s_20step"
    )
    (dst / f"{mode}_15s_20step.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2)
    )
PY

for spec in "i2v 8192" "flf 8193"; do
  set -- $spec
  mode=$1
  port=$2
  setsid python \
    /path/to/submit_h3_prompt.py \
    --workflow "$BASE/${mode}_15s_20step.json" \
    --label "single_baseline_${mode}_15s_20step" \
    --port "$port" \
    --timeout 1800 \
    --runs-dir "$BASE/$mode-evidence" \
    >"$BASE/$mode-submit.log" 2>&1 < /dev/null &
  echo $! >"$BASE/$mode-submit.pid"
done

printf 'STARTED i2v_server=%s flf_server=%s i2v_submit=%s flf_submit=%s\n' \
  "$(cat "$BASE/i2v-server.pid")" \
  "$(cat "$BASE/flf-server.pid")" \
  "$(cat "$BASE/i2v-submit.pid")" \
  "$(cat "$BASE/flf-submit.pid")"
