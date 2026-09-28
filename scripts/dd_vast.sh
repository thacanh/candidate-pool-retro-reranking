#!/usr/bin/env bash
# Vast launch/status only; scientific settings remain in the frozen runners.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
JOB="${1:-}"
case "$JOB" in
  N3) DEFAULT_WORKERS=6; RUNNER=scripts/run_dd_n3_remote.sh ;;
  N4) DEFAULT_WORKERS=8; RUNNER=scripts/run_dd_n4_remote.sh ;;
  *) echo 'Usage: bash scripts/dd_vast.sh N3|N4 [status|resume]'; exit 2 ;;
esac
SESSION="dd-${JOB,,}"
LOGDIR="$ROOT/logs/dd_final_round"
MASTER="$LOGDIR/${JOB}_master.log"
MODE="${2:-start}"

if [[ "$MODE" == status ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "$JOB session: running"
  else
    echo "$JOB session: not running"
  fi
  if [[ -f "$LOGDIR/${JOB}.exit" ]]; then
    printf 'Exit code: '; cat "$LOGDIR/${JOB}.exit"
  fi
  if [[ "$JOB" == N3 ]]; then
    for stage in N3_S1 N3_S2 N3_S3; do
      if [[ -f "/workspace/dd_n3_archives/$stage.receipt.json" ]]; then
        echo "$stage: archived and verified"
      elif [[ -f "outputs/dd_final_round/$stage/manifest.json" ]]; then
        echo "$stage: manifest exists; check archive log"
      else
        echo "$stage: pending or running"
      fi
    done
  fi
  if [[ -d "outputs/dd_final_round/$JOB" ]]; then
    printf 'Trial files: '
    find "outputs/dd_final_round/$JOB" -type f -name trial.json | wc -l
    if [[ "$JOB" == N4 ]]; then
      for pool in aizynthfinder_only localretro_only merged; do
        printf '%s Morgan shards: ' "$pool"
        { find "outputs/dd_final_round/N4/$pool/features/shards" -maxdepth 1 -name 'shard_*.json' 2>/dev/null || true; } | wc -l
      done
    fi
  elif [[ "$JOB" == N3 ]]; then
    for stage in N3_S2 N3_S3; do
      printf '%s trial files: ' "$stage"
      { find "outputs/dd_final_round/$stage" -type f -name trial.json 2>/dev/null || true; } | wc -l
    done
  fi
  tail -n 8 "$MASTER" 2>/dev/null || true
  for stage_log in "$LOGDIR/${JOB}_run.log" "$LOGDIR/${JOB}_S1.log" "$LOGDIR/${JOB}_S2.log" "$LOGDIR/${JOB}_S3.log"; do
    if [[ -f "$stage_log" ]]; then
      echo "--- $(basename "$stage_log") ---"
      tail -n 2 "$stage_log"
    fi
  done
  exit 0
fi

export DD_FIT_WORKERS="${DD_FIT_WORKERS:-$DEFAULT_WORKERS}"
export DD_FEATURE_WORKERS="${DD_FEATURE_WORKERS:-8}"
export DD_PYTHON=/workspace/dd-final-py310/bin/python
export PYTHONUNBUFFERED=1

if [[ "$MODE" == _run || "$MODE" == _resume ]]; then
  mkdir -p "$LOGDIR"
  exec >> "$MASTER" 2>&1
  trap 'rc=$?; printf "%s\n" "$rc" > "$LOGDIR/${JOB}.exit"; echo "JOB_EXIT_CODE=$rc"' EXIT
  echo "$JOB started at $(date -u +%FT%TZ); fit_workers=$DD_FIT_WORKERS; feature_workers=$DD_FEATURE_WORKERS"
  if [[ "$MODE" == _run ]]; then
    bash scripts/dd_setup_remote.sh
  else
    [[ -x "$DD_PYTHON" ]] || { echo 'Existing environment missing; cannot resume.'; exit 1; }
  fi
  bash "$RUNNER"
  exit 0
fi

[[ "$MODE" == start || ( "$MODE" == resume && "$JOB" == N3 ) ]] || { echo "Unknown/unsupported mode: $MODE"; exit 2; }
command -v tmux >/dev/null || { echo 'tmux is missing from the selected image.'; exit 1; }
command -v uv >/dev/null || { echo 'uv is missing; use the Vast PyTorch/Jupyter image with uv.'; exit 1; }
if [[ ( "$MODE" == start && -e "$MASTER" ) ]] || tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Existing $JOB run detected. Use: bash scripts/dd_vast.sh $JOB status"
  exit 1
fi
mkdir -p "$LOGDIR"
INNER=_run
[[ "$MODE" == resume ]] && INNER=_resume
if [[ -f "$LOGDIR/${JOB}.exit" ]]; then
  mv -- "$LOGDIR/${JOB}.exit" "$LOGDIR/${JOB}.exit.$(date -u +%Y%m%dT%H%M%S)"
fi
printf -v COMMAND '%q ' env "DD_FIT_WORKERS=$DD_FIT_WORKERS" "DD_FEATURE_WORKERS=$DD_FEATURE_WORKERS" bash "$ROOT/scripts/dd_vast.sh" "$JOB" "$INNER"
tmux new-session -d -s "$SESSION" "$COMMAND"
echo "$JOB launched in tmux session $SESSION. Setup and science logs: $LOGDIR"
echo "Track: watch -n 20 bash scripts/dd_vast.sh $JOB status"
