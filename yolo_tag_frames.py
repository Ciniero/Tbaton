import json
from pathlib import Path
from ultralytics import YOLO

# --- NEW: locate model path centrally ---
MODEL_PATH = Path("/Users/ale/tbaton_base/models/hand_detector.pt")
if not MODEL_PATH.exists():
    MODEL_PATH = Path("hand_detector.pt")
# ----------------------------------------

FRAMES_DIR = Path("frames")
OUT_JSON   = Path("frame_detections.json")

TOOLISH = {"wrench", "screwdriver", "tool", "plier"}
HANDISH = {"hand", "right hand", "left hand"}

def main():
    model = YOLO(str(MODEL_PATH))
    results = []

    frame_files = sorted(FRAMES_DIR.glob("frame_*.jpg"))
    if not frame_files:
        raise SystemExit("No frames found in ./frames — run extract_keyframes.py first")

    for f in frame_files:
        ts = float(f.stem.split("_")[1])
        r = model.predict(source=str(f), imgsz=640, conf=0.25, verbose=False)[0]
        dets = [r.names[int(b.cls[0])] for b in r.boxes]

        has_hand = any(n.lower() in HANDISH for n in dets)
        has_tool = any(n.lower() in TOOLISH for n in dets)

        results.append({
            "frame": f.name,
            "t_sec": ts,
            "detections": dets,
            "has_hand": has_hand,
            "has_tool": has_tool,
        })

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"✅ wrote {OUT_JSON} with {len(results)} frames")

if __name__ == "__main__":
    main()
