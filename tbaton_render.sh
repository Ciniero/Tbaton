#!/bin/bash
# =============================================================
#  T-Baton: Headless export (captions + TTS, no live playback)
#  Uses: tbaton_player_tts_headless.py
#  Output:
#    *_captioned_silent.mp4   (video with captions, no audio)
#    *_tts.m4a                (TTS audio-only)
#    *_captioned_TTS.mp4      (final muxed)
# =============================================================

SCRIPTS="$HOME/tbaton_base/scripts"

set -e

# --- Resolve config ---
CONF_FILE="tbaton.conf"
if [ -f "$CONF_FILE" ]; then
  source "$CONF_FILE"
fi

VIDEO_PATH="${VIDEO:-demo.mp4}"
STEPS_PATH="${STEPS:-$(basename "$VIDEO_PATH" .mp4)_steps_auto.json}"
TTS_DIR="${TTS_DIR:-tts_audio}"

# --- Script path ---

HEADLESS_PY="$SCRIPTS/tbaton_player_tts_headless.py"

if [ ! -f "$HEADLESS_PY" ]; then
  echo "[ERROR] tbaton_player_tts_headless.py not found in base folder."
  exit 1
fi

# --- Activate tbenv_clean (conda) if available ---
if command -v conda >/dev/null 2>&1; then
  if conda env list 2>/dev/null | grep -q "tbenv_clean"; then
    echo "[INFO] Activating conda environment: tbenv_clean"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate tbenv_clean
  fi
fi

# --- Ensure Python deps (safe to re-run) ---
pip install -q imageio-ffmpeg opencv-python

# --- Run headless export ---
echo "=============================================="
echo " T-Baton Headless Export"
echo "----------------------------------------------"
echo " Video:   $VIDEO_PATH"
echo " Steps:   $STEPS_PATH"
echo " TTS dir: $TTS_DIR"
echo "----------------------------------------------"

python "$HEADLESS_PY" \
  --video "$VIDEO_PATH" \
  --steps "$STEPS_PATH" \
  --tts_dir "$TTS_DIR" \
  "$@"

echo "=============================================="
echo "✅ Done."
