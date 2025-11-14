#!/usr/bin/env python3
"""
tbaton_player_tts.py

Player for T-Baton MVP:
- show captions over video
- play per-step TTS from tts_audio/step_XXX.wav
- IMPORTANT: do NOT interrupt an ongoing TTS just because the next visual segment started
"""

import cv2
import json
import subprocess
import argparse
from pathlib import Path

NUDGE_STEP = 0.25  # seconds


def load_steps(path: Path):
    data = json.loads(path.read_text())
    return sorted(data, key=lambda s: s["t_start"])


def find_step_at(steps, t, offset):
    t_adj = t + offset
    for s in steps:
        if s["t_start"] <= t_adj <= s["t_end"]:
            return s
    return None


def wrap(text, max_chars=70):
    words = text.split()
    lines = []
    cur = []
    count = 0
    for w in words:
        extra = 1 if cur else 0
        if count + len(w) + extra > max_chars:
            lines.append(" ".join(cur))
            cur = [w]
            count = len(w)
        else:
            cur.append(w)
            count += len(w) + extra
    if cur:
        lines.append(" ".join(cur))
    return lines


def draw_caption(frame, text, offset):
    if not text:
        return frame
    out = frame.copy()
    h, w, _ = out.shape
    margin = 10
    line_h = 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thick = 2

    if offset:
        text = f"{text}  (off {offset:+.2f}s)"

    lines = wrap(text, 70)
    box_h = margin * 2 + line_h * len(lines)
    y1 = h - box_h
    y2 = h

    overlay = out.copy()
    cv2.rectangle(overlay, (0, y1), (w, y2), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)

    y = y1 + margin + line_h - 6
    for ln in lines:
        cv2.putText(out, ln, (margin, y), font, font_scale, (255, 255, 255), thick, cv2.LINE_AA)
        y += line_h

    return out


def stop_tts(proc_holder):
    proc = proc_holder.get("proc")
    if proc and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass
    proc_holder["proc"] = None


def play_tts_for(step, tts_dir: Path, proc_holder):
    """Start TTS for a step (assumes caller already checked that no TTS is playing)."""
    sid = step.get("step_id", 0)
    wav = tts_dir / f"step_{sid:03d}.wav"
    if not wav.exists():
        return None
    proc = subprocess.Popen(["afplay", str(wav)])
    proc_holder["proc"] = proc
    return proc


def main():
    parser = argparse.ArgumentParser(description="T-Baton caption+TTS player")
    parser.add_argument("--video", required=True, help="video file (e.g. demo.mp4)")
    parser.add_argument("--steps", required=True, help="steps json (e.g. demo_steps_auto.json)")
    parser.add_argument("--tts_dir", default="tts_audio", help="folder with step_XXX.wav")
    args = parser.parse_args()

    video_path = Path(args.video)
    steps_path = Path(args.steps)
    tts_dir = Path(args.tts_dir)

    if not video_path.exists():
        raise SystemExit(f"[ERROR] video not found: {video_path}")
    if not steps_path.exists():
        raise SystemExit(f"[ERROR] steps json not found: {steps_path}")

    steps = load_steps(steps_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] cannot open video {video_path}")

    print("Controls: [ earlier, ] later, \\ reset, q quit")

    offset = 0.0

    # which step we last *successfully started* (with TTS)
    last_started_step_id = None
    last_started_step_obj = None

    # 👇 when TTS is playing, we lock to this step
    locked_step_id = None

    proc_holder = {"proc": None}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # is a TTS currently running?
        proc = proc_holder.get("proc")
        tts_playing = proc is not None and proc.poll() is None

        if tts_playing and locked_step_id is not None:
            # stay on the locked step — do NOT start new audio, do NOT change caption
            step = next((s for s in steps if s["step_id"] == locked_step_id), None)
            caption = step["instruction"] if step else ""
        else:
            # no audio playing -> we are allowed to follow video time
            step = find_step_at(steps, t, offset)
            if step:
                caption = step["instruction"]
                if step["step_id"] != last_started_step_id:
                    # only start audio if nothing is playing right now
                    proc = play_tts_for(step, tts_dir, proc_holder)
                    if proc is not None:
                        locked_step_id = step["step_id"]
                    last_started_step_id = step["step_id"]
                    last_started_step_obj = step
            else:
                caption = ""
                locked_step_id = None

        # if we were locked but audio finished this frame -> unlock
        proc = proc_holder.get("proc")
        if locked_step_id is not None and (proc is None or proc.poll() is not None):
            locked_step_id = None

        frame = draw_caption(frame, caption, offset)
        cv2.imshow("T-Baton Player", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord(']'):
            offset += NUDGE_STEP
        elif k == ord('['):
            offset -= NUDGE_STEP
        elif k == ord('\\'):
            offset = 0.0

    stop_tts(proc_holder)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
