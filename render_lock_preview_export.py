#!/usr/bin/env python3
"""
Render a muted, captioned video with TTS mixed at exact trigger times
derived from video frame time (export == preview logic).

Fix: If OpenCV VideoWriter can't open MP4 (macOS codec issue),
we fall back to an FFmpeg rawvideo pipe so writing never fails.
"""

import argparse, json, os, shutil, subprocess, contextlib, wave, sys
from pathlib import Path
import cv2
import numpy as np

TTS_EXTS = (".wav", ".m4a", ".mp3")

# Optional: bundled ffmpeg via imageio-ffmpeg
try:
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get("PATH","")
except Exception:
    pass

def ensure_ffmpeg():
    if not shutil.which("ffmpeg"):
        raise SystemExit("[ERROR] ffmpeg not found on PATH. Install ffmpeg or `pip install imageio-ffmpeg`.")

def run(cmd, tag):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print(p.stdout)
        raise SystemExit(f"[ERROR] {tag} failed.")
    return p.stdout

def load_steps(path: Path):
    data = json.loads(path.read_text())
    return sorted(data, key=lambda s: s["t_start"])

def find_step_audio(tts_dir: Path, step_id: int):
    for ext in TTS_EXTS:
        p = tts_dir / f"step_{step_id:03d}{ext}"
        if p.exists(): return p
    return None

def ensure_pcm_wav(src: Path, dst: Path, ar="48000", ac="1"):
    cmd = ["ffmpeg","-y","-i",str(src),"-ar",ar,"-ac",ac,"-c:a","pcm_s16le",str(dst)]
    run(cmd, f"re-encode {src.name} -> PCM WAV")

def wav_duration_sec(path: Path) -> float:
    with contextlib.closing(wave.open(str(path), "rb")) as w:
        return w.getnframes() / float(w.getframerate())

def wrap(text, max_chars=70):
    words = text.split()
    out, cur, count = [], [], 0
    for w in words:
        extra = 1 if cur else 0
        if count + len(w) + extra > max_chars:
            out.append(" ".join(cur)); cur, count = [w], len(w)
        else:
            cur.append(w); count += len(w) + extra
    if cur: out.append(" ".join(cur))
    return out

def draw_caption(frame, text):
    if not text: return frame
    out = frame.copy()
    h, w = out.shape[:2]
    margin, line_h = 10, 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thick = 0.6, 2
    lines = wrap(text, 70)
    box_h = margin*2 + line_h*len(lines)
    y1, y2 = h - box_h, h
    overlay = out.copy()
    cv2.rectangle(overlay, (0, y1), (w, y2), (0,0,0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
    y = y1 + margin + line_h - 6
    for ln in lines:
        cv2.putText(out, ln, (margin, y), font, font_scale, (255,255,255), thick, cv2.LINE_AA)
        y += line_h
    return out

def find_step_at_time(steps, t_logic, offset):
    t_adj = t_logic + offset
    for s in steps:
        if s["t_start"] <= t_adj <= s["t_end"]:
            return s
    return None

def build_audio_from_events(tts_events, out_path: Path):
    """Mix narration from exact trigger events using per-input adelay; output audio-only file."""
    ensure_ffmpeg()
    tts_events = sorted(tts_events, key=lambda e: e[1])
    inputs, filters, labels = [], [], []
    for idx, (sid, t0, wav_path) in enumerate(tts_events):
        if not Path(wav_path).exists():
            continue
        delay_ms = max(0, int(round(t0 * 1000)))
        inputs += ["-i", str(wav_path)]
        lbl = f"a{idx}"
        filters.append(f"[{idx}:a]adelay={delay_ms}|{delay_ms}:all=1[{lbl}]")
        labels.append(f"[{lbl}]")
    if not labels:
        print("[WARN] No valid WAV inputs to mix.")
        return False

    amix = f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0[mix]"
    filter_complex = ";".join(filters + [amix])

    cmd = [
        "ffmpeg","-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map","[mix]",
        "-c:a","aac",
        "-movflags","+faststart",
        "-f","mp4",     # m4a-compatible mp4 container
        str(out_path)
    ]
    print("[FFMPEG] mix (events):", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print("[WARN] ffmpeg mix failed.")
        return False

def mux_video_audio(video_noaudio: Path, audio_in: Path, final_out: Path):
    ensure_ffmpeg()
    cmd = [
        "ffmpeg","-y",
        "-i", str(video_noaudio),
        "-i", str(audio_in),
        "-map","0:v:0?","-map","1:a:0?",
        "-c:v","copy","-c:a","aac",
        "-shortest",
        "-movflags","+faststart",
        str(final_out)
    ]
    print("[FFMPEG] mux:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print("[WARN] ffmpeg mux failed.")
        return False

# -------- FFmpeg pipe writer (fallback when cv2.VideoWriter fails) ----------
class FFmpegPipeWriter:
    def __init__(self, out_path: Path, fps: float, width: int, height: int, crf="20", preset="veryfast"):
        ensure_ffmpeg()
        self.out_path = str(out_path)
        self.w, self.h = width, height
        self.proc = subprocess.Popen(
            [
                "ffmpeg","-y",
                "-f","rawvideo",
                "-vcodec","rawvideo",
                "-pix_fmt","bgr24",
                "-s", f"{self.w}x{self.h}",
                "-r", f"{fps}",
                "-i","-",
                "-an",
                "-vcodec","libx264",
                "-pix_fmt","yuv420p",
                "-preset", preset,
                "-crf", crf,
                "-movflags","+faststart",
                self.out_path
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    def write(self, frame):
        # frame must be HxWx3 BGR uint8
        if self.proc and self.proc.stdin:
            self.proc.stdin.write(frame.tobytes())

    def release(self):
        if self.proc:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait()
            self.proc = None

def main():
    ap = argparse.ArgumentParser(description="Frame-driven caption+TTS renderer (export == preview logic).")
    ap.add_argument("--video", required=True, help="input video (mp4)")
    ap.add_argument("--steps", required=True, help="steps json")
    ap.add_argument("--tts_dir", default="tts_audio", help="folder with step_XXX.(wav|m4a|mp3)")
    ap.add_argument("--offset", type=float, default=0.0, help="global timing nudge in seconds (+later, -earlier)")
    args = ap.parse_args()

    video_path = Path(args.video)
    steps_path = Path(args.steps)
    tts_dir = Path(args.tts_dir)
    if not video_path.exists(): raise SystemExit(f"[ERROR] video not found: {video_path}")
    if not steps_path.exists(): raise SystemExit(f"[ERROR] steps json not found: {steps_path}")
    steps = load_steps(steps_path)

    # Prepare per-step clean WAV + durations
    cache = tts_dir / "_pcm_tmp"; cache.mkdir(parents=True, exist_ok=True)
    step_wavs = {}   # sid -> (wav_pcm_path, dur)
    for s in steps:
        sid = int(s.get("step_id", 0))
        a = find_step_audio(tts_dir, sid)
        if not a: continue
        dst = cache / f"step_{sid:03d}.wav"
        ensure_pcm_wav(a, dst, ar="48000", ac="1")
        d = wav_duration_sec(dst)
        if d > 0.01:
            step_wavs[sid] = (dst, d)

    if not step_wavs:
        raise SystemExit("[ERROR] No valid TTS per-step audio found.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise SystemExit(f"[ERROR] cannot open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps < 1: fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Try OpenCV writer first
    out_video_noaudio = video_path.with_name(video_path.stem + "_captioned_silent.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_video_noaudio), fourcc, fps, (w, h))

    use_pipe = False
    if not writer.isOpened():
        # Fallback to FFmpeg pipe
        print(f"[WARN] OpenCV VideoWriter failed. Falling back to FFmpeg pipe for {out_video_noaudio}")
        writer = FFmpegPipeWriter(out_video_noaudio, fps, w, h, crf="20", preset="veryfast")
        use_pipe = True

    print(f"[INFO] Writing (muted) captioned video to: {out_video_noaudio}")

    locked_sid = None
    lock_ends_at = -1.0
    last_started_sid = None
    tts_events = []  # (sid, trigger_time, wav_pcm_path)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break

        t_logic = frame_idx / float(fps)

        # Update lock state
        if locked_sid is not None and t_logic >= lock_ends_at:
            locked_sid = None

        if locked_sid is None:
            step = find_step_at_time(steps, t_logic, args.offset)
            if step:
                sid = int(step["step_id"])
                if sid != last_started_sid and sid in step_wavs:
                    wav_path, dur = step_wavs[sid]
                    locked_sid = sid
                    lock_ends_at = t_logic + dur
                    last_started_sid = sid
                    tts_events.append((sid, t_logic, wav_path))

        caption = ""
        if locked_sid is not None:
            step = next((s for s in steps if int(s["step_id"]) == locked_sid), None)
            caption = (step.get("instruction") or "") if step else ""

        frame = draw_caption(frame, caption)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    if use_pipe:
        writer.release()
    else:
        writer.release()

    print(f"[INFO] Saved captioned (silent) video: {out_video_noaudio}")

    # Build audio and mux
    tts_audio = video_path.with_name(video_path.stem + "_tts.m4a")
    ok_mix = build_audio_from_events(tts_events, tts_audio)
    if ok_mix:
        final_out = video_path.with_name(video_path.stem + "_captioned_TTS.mp4")
        ok_mux = mux_video_audio(out_video_noaudio, tts_audio, final_out)
        if ok_mux:
            print(f"✅ Final export (preview-matched timing): {final_out}")
        else:
            print("⚠️ Mux failed. You still have the captioned (silent) video and the TTS audio.")
    else:
        print("⚠️ Audio mix failed. You still have the captioned (silent) video.")

if __name__ == "__main__":
    ensure_ffmpeg()
    main()
