import cv2, csv, os
from pathlib import Path

# --- NEW: read video from local config ---
CONF = Path("tbaton.conf")
VIDEO = "demo.mp4"
if CONF.exists():
    for line in CONF.read_text().splitlines():
        if line.startswith("VIDEO="):
            VIDEO = line.split("=", 1)[1].strip()
# ----------------------------------------

OUT_DIR = Path("frames")
CSV_PATH = Path("motion_timeline.csv")
STEP = 0.5  # seconds between samples (adjust 0.25–1.0 as you like)

def main():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Cannot open {VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = frame_count / fps

    OUT_DIR.mkdir(exist_ok=True)
    prev_gray = None

    rows = [("t_sec","motion")]
    t = 0.0
    while t <= dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t*1000.0)
        ok, frame = cap.read()
        if not ok: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (640, int(gray.shape[0]*640/gray.shape[1])))

        if prev_gray is None:
            motion = 0.0
        else:
            diff = cv2.absdiff(gray, prev_gray)
            motion = float(diff.mean())

        prev_gray = gray
        fname = OUT_DIR / f"frame_{t:07.2f}.jpg"
        cv2.imwrite(str(fname), frame)
        rows.append((round(t,2), round(motion,6)))
        t += STEP

    cap.release()
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"✅ Saved {len(rows)-1} samples to {CSV_PATH} and frames to {OUT_DIR}/")

if __name__ == "__main__":
    main()
