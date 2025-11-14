#!/bin/bash
# T-Baton: play-only (robust)

set -u

# 0) activate central venv + ffmpeg
if [ -f "$HOME/tbaton_base/tbenv_clean/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$HOME/tbaton_base/tbenv_clean/bin/activate"
else
  echo "[ERROR] Cannot find venv at $HOME/tbaton_base/tbenv_clean/bin/activate"
  exit 1
fi
export PATH="$HOME/local/bin:$PATH"

SCRIPTS="$HOME/tbaton_base/scripts"
PLAYER="$SCRIPTS/tbaton_player_tts.py"

# 1) defaults, then override from local config if present
VIDEO="demo.mp4"
STEPS="demo_steps_auto.json"
if [ -f ./tbaton.conf ]; then
  # strip inline comments and blank lines
  while IFS= read -r raw; do
    line="${raw%%#*}"
    line="$(echo "$line" | awk '{$1=$1};1')"
    [ -z "$line" ] && continue
    case "$line" in
      VIDEO=*)  VIDEO="${line#VIDEO=}" ;;
      STEPS=*)  STEPS="${line#STEPS=}" ;;
    esac
  done < ./tbaton.conf
fi

echo "[tbaton_play] cwd: $(pwd)"
echo "[tbaton_play] video: $VIDEO"
echo "[tbaton_play] steps: $STEPS"
echo "[tbaton_play] player: $PLAYER"

# 2) sanity checks
if [ ! -f "$PLAYER" ]; then
  echo "[ERROR] Player script not found at: $PLAYER"
  exit 1
fi

if [ ! -f "$VIDEO" ]; then
  echo "[ERROR] Video not found in this folder: $VIDEO"
  echo "        Put the file in $(pwd) or set VIDEO=... in tbaton.conf"
  exit 1
fi

if [ ! -f "$STEPS" ]; then
  echo "[ERROR] Steps JSON not found: $STEPS"
  echo "        Run the build steps to generate it, or set STEPS=... in tbaton.conf"
  exit 1
fi

# 3) run
if command -v python3 >/dev/null 2>&1; then
  python3 "$PLAYER" --video "$VIDEO" --steps "$STEPS"
else
  python "$PLAYER" --video "$VIDEO" --steps "$STEPS"
fi
