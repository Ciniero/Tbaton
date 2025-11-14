import json, pyttsx3
from pathlib import Path

CONF = Path("tbaton.conf")
VIDEO = "demo.mp4"
if CONF.exists():
    for line in CONF.read_text().splitlines():
        if line.startswith("VIDEO="):
            VIDEO = line.split("=", 1)[1].strip()
STEPS_JSON = Path(f"{Path(VIDEO).stem}_steps_auto.json")

OUT_DIR = Path("tts_audio"); OUT_DIR.mkdir(exist_ok=True)
engine = pyttsx3.init()
rate = engine.getProperty("rate")
engine.setProperty("rate", int(rate * 0.85))

def main():
    steps = json.loads(STEPS_JSON.read_text())
    for step in steps:
        sid = step.get("step_id", 0)
        text = step.get("instruction", "Perform the next operation.")
        wav_path = OUT_DIR / f"step_{sid:03d}.wav"
        print(f"[TTS] {sid}: {text}")
        engine.save_to_file(text, str(wav_path))
    engine.runAndWait()
    print("✅ done generating TTS")

if __name__ == "__main__":
    main()
