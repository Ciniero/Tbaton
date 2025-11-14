import json
from pathlib import Path

DETECTIONS_JSON = Path("frame_detections.json")
OUT_JSON = Path("visual_segments.json")

MIN_SEG_SEC = 5.0
MAX_SEG_SEC = 8.0
STABILITY_WINDOW = 2.5
MAX_GAP_SEC = 1.0
COOLDOWN_SEC = 1.0

def normalize_set(dets):
    names = []
    for d in dets:
        if isinstance(d, str):
            names.append(d.lower())
        elif isinstance(d, dict):
            cls_name = d.get("cls") or d.get("name") or d.get("class")
            if cls_name:
                names.append(str(cls_name).lower())
    buckets = []
    for n in names:
        if "hand" in n:
            buckets.append("hand")
        elif "wrench" in n or "screwdriver" in n or "tool" in n:
            buckets.append("tool")
        else:
            buckets.append(n)
    return sorted(set(buckets))

def load_detections():
    if not DETECTIONS_JSON.exists():
        raise SystemExit(f"[ERROR] {DETECTIONS_JSON} not found. Run yolo_tag_frames.py first.")
    data = json.loads(DETECTIONS_JSON.read_text())
    return sorted(data, key=lambda x: x["t_sec"])

def same_ctx(a, b): return set(a or []) == set(b or [])

def make_segments(data):
    segments = []
    first = data[0]
    cur_objs = normalize_set(first["detections"])
    cur_seg = {"t_start": first["t_sec"], "t_end": first["t_sec"], "objects": cur_objs}
    last_t = first["t_sec"]; last_objs = cur_objs; last_seg_start = first["t_sec"]

    def change_is_stable(idx, prev_objs):
        new_objs = normalize_set(data[idx]["detections"])
        start_t = data[idx]["t_sec"]
        for j in range(idx + 1, len(data)):
            tj = data[j]["t_sec"]
            objs_j = normalize_set(data[j]["detections"])
            if tj - start_t > STABILITY_WINDOW:
                return True
            if same_ctx(objs_j, prev_objs):
                return False
        return True

    for idx in range(1, len(data)):
        item = data[idx]; t = item["t_sec"]; objs = normalize_set(item["detections"])
        big_gap = (t - last_t) > MAX_GAP_SEC
        changed = not same_ctx(objs, last_objs)
        in_cooldown = (t - last_seg_start) < COOLDOWN_SEC
        cur_len = t - cur_seg["t_start"]; force_cut = cur_len >= MAX_SEG_SEC

        if big_gap or force_cut or (changed and not in_cooldown and change_is_stable(idx, last_objs)):
            segments.append(cur_seg)
            cur_seg = {"t_start": t, "t_end": t, "objects": objs}
            last_seg_start = t
        else:
            cur_seg["t_end"] = t
        last_t = t; last_objs = objs

    segments.append(cur_seg)

    final_segments = []
    for seg in segments:
        dur = seg["t_end"] - seg["t_start"]
        if dur >= MIN_SEG_SEC or not final_segments:
            final_segments.append(seg)
        else:
            prev = final_segments[-1]
            prev["t_end"] = seg["t_end"]
            prev["objects"] = sorted(set(prev["objects"] + seg["objects"]))

    return final_segments

def main():
    data = load_detections()
    segs = make_segments(data)
    OUT_JSON.write_text(json.dumps(segs, indent=2))
    print(f"✅ wrote {OUT_JSON} with {len(segs)} segments")

if __name__ == "__main__":
    main()
