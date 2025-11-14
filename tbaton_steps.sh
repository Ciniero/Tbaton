#!/bin/bash
# T-Baton: build-only (NO player)

# 0) env
source "$HOME/tbaton_base/tbenv_clean/bin/activate"
export PATH="$HOME/local/bin:$PATH"

SCRIPTS="$HOME/tbaton_base/scripts"

# 1) frames
python "$SCRIPTS/extract_keyframes.py" || exit 1

# 2) yolo
python "$SCRIPTS/yolo_tag_frames.py" || exit 1

# 3) visual segments
python "$SCRIPTS/make_visual_segments.py" || exit 1

# 4) load video name from tbaton.conf (or fallback)
VIDEO="demo.mp4"
WHISPER_DIR="demo_whisper"
STEPS="demo_steps_auto.json"
if [ -f ./tbaton.conf ]; then
  source ./tbaton.conf
fi

# 4) whisper
whisper "$VIDEO" --model medium --task transcribe --output_format json --verbose True -o "$WHISPER_DIR" || exit 1

# 5) fuse
python "$SCRIPTS/fuse_visual_whisper.py" || exit 1

# 6) tts
python "$SCRIPTS/tts_generate_from_steps.py" || exit 1

echo
echo "✅ Build done."
echo "➡ Now run the player separately:"
echo "   python $SCRIPTS/tbaton_player_tts.py --video \"$VIDEO\" --steps \"$STEPS\""
