import json
import re
from pathlib import Path

# -------------------- NEW: robust path resolution (minimal) --------------------
# Reads VIDEO / WHISPER_DIR / STEPS from ./tbaton.conf if present.
CONF = Path("tbaton.conf")
VIDEO = "demo.mp4"
WHISPER_DIR = None
STEPS = None

if CONF.exists():
    for line in CONF.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip().upper()
            v = v.strip()
            if k == "VIDEO":
                VIDEO = v
            elif k == "WHISPER_DIR":
                WHISPER_DIR = v
            elif k == "STEPS":
                STEPS = v

stem = Path(VIDEO).stem

# Inputs/outputs (defaults preserved if no conf is provided)
VIS_SEG = Path("visual_segments.json")                                  # from make_visual_segments.py
OUT     = Path(STEPS) if STEPS else Path(f"{stem}_steps_auto.json")     # final steps JSON

# Resolve Whisper transcript path:
def _resolve_whisper_json():
    wdir = Path(WHISPER_DIR) if WHISPER_DIR else Path(f"{stem}_whisper")
    # original default would have been: demo_whisper/demo.json
    preferred = wdir / f"{stem}.json"
    if preferred.exists():
        return preferred
    # tolerate common typo (e.g., pump.jason)
    typo = wdir / f"{stem}.jason"
    if typo.exists():
        return typo
    # fallback: any *.json / *.jason in the folder (pick newest)
    cands = list(wdir.glob("*.json")) + list(wdir.glob("*.jason"))
    if cands:
        return max(cands, key=lambda p: p.stat().st_mtime)
    # if nothing resolves, keep original-style default to trigger the same error path
    return Path("demo_whisper/demo.json")

WHISP = _resolve_whisper_json()
# -------------------- END NEW --------------------------------------------------


def load_visual_segments():
    if not VIS_SEG.exists():
        raise SystemExit(f"[ERROR] {VIS_SEG} not found. Run make_visual_segments.py first.")
    return json.loads(VIS_SEG.read_text())


def load_whisper_segments():
    if not WHISP.exists():
        raise SystemExit(f"[ERROR] {WHISP} not found. Run whisper on the video first.")
    data = json.loads(WHISP.read_text())
    return data.get("segments", [])


def best_whisper_for(vis_seg, whisper_segments, min_overlap=0.10):
    """
    Pick the whisper segment that overlaps most with this visual segment.
    Return (segment or None, overlap_seconds)
    """
    best = None
    best_ov = 0.0
    for ws in whisper_segments:
        a = max(vis_seg["t_start"], ws["start"])
        b = min(vis_seg["t_end"], ws["end"])
        ov = b - a
        if ov > best_ov:
            best_ov = ov
            best = ws
    if best is None or best_ov < min_overlap:
        return None, 0.0
    return best, best_ov


def clean_instruction(txt: str) -> str:
    """Turn raw ASR text into a simple training-style sentence."""
    txt = txt.strip()

    # remove filler at start
    txt = re.sub(r"^(now|then|next|okay|alright|so)[, ]+", "", txt, flags=re.I)

    # turn "I'm going to" → imperative
    txt = re.sub(r"(?i)i(?:'m| am)\s+going to\s+", "", txt)
    txt = re.sub(r"(?i)we(?:'re| are)\s+going to\s+", "", txt)

    # capitalize
    if txt and not txt[0].isupper():
        txt = txt[0].upper() + txt[1:]

    # ensure period
    if txt and not txt.endswith("."):
        txt += "."

    return txt


def template_from_objects(objs):
    """Fallback when no speech overlaps this visual segment."""
    objs = [o.lower() for o in (objs or [])]

    if not objs:
        return "Perform the next operation."

    if "hand" in objs and len(objs) == 1:
        return "Adjust or place the component carefully."

    if "wrench" in objs or "screwdriver" in objs or "tool" in objs:
        return "Tighten or loosen the fastener using the tool."

    if "spring" in objs or "retainer" in objs:
        return "Install the spring retainer and ensure it is fully seated."

    # generic
    return "Handle: " + ", ".join(sorted(set(objs))) + "."


def main():
    vis_segments = load_visual_segments()
    whisper_segments = load_whisper_segments()

    steps = []
    step_id = 1

    for vs in vis_segments:
        ws, ov = best_whisper_for(vs, whisper_segments, min_overlap=0.10)
        if ws:
            instr = clean_instruction(ws["text"])
        else:
            instr = template_from_objects(vs.get("objects"))

        steps.append({
            "step_id": step_id,
            "t_start": round(vs["t_start"], 2),
            "t_end": round(vs["t_end"], 2),
            "instruction": instr,
            # keep objects so we can highlight them later in AR
            "objects": vs.get("objects", [])
        })
        step_id += 1

    OUT.write_text(json.dumps(steps, indent=2, ensure_ascii=False))
    print(f"✅ wrote {OUT} with {len(steps)} auto steps")
    if steps:
        print("sample:", steps[:3])


if __name__ == "__main__":
    main()
