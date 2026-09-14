#!/usr/bin/env bash
# watchdog_gen.sh - keep a batch generation driver alive until expected outputs exist.
#
# Usage:
#   watchdog_gen.sh EXPECTED OUT_DIR LOG_FILE NAME_PATTERN PID_FILE -- COMMAND...
#
# Uses a PID file (not pgrep) so it never confuses itself with the child.
set -u

EXPECTED="$1"; OUT_DIR="$2"; LOG="$3"; NAME_PATTERN="$4"; PID_FILE="$5"; shift 5
[ "${1:-}" = "--" ] && shift

mkdir -p "$(dirname "$LOG")" "$(dirname "$PID_FILE")" "$OUT_DIR"
echo "watchdog start: expected=$EXPECTED out=$OUT_DIR name=$NAME_PATTERN pid=$PID_FILE" >> "$LOG"

while true; do
  n=$(find "$OUT_DIR" -name "$NAME_PATTERN" 2>/dev/null | wc -l)
  if [ "$n" -ge "$EXPECTED" ]; then
    echo "watchdog: complete $n/$EXPECTED" >> "$LOG"
    exit 0
  fi
  alive=0
  if [ -f "$PID_FILE" ]; then
    pid=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then alive=1; fi
  fi
  if [ "$alive" -eq 0 ]; then
    echo "watchdog: restart at $n/$EXPECTED -> $*" >> "$LOG"
    setsid nohup "$@" >> "$LOG" 2>&1 < /dev/null &
    echo $! > "$PID_FILE"
    sleep 10
  fi
  sleep 60
done
