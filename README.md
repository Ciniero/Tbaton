# Video to Step-by-Step Instructions Generator

This tool analyzes silent video recordings and automatically generates detailed step-by-step text instructions describing the actions performed in the video.

## Features

- **Object Detection**: Uses YOLOv8 or OpenCV DNN to detect objects in video frames
- **Action Recognition**: Identifies common activities based on object interactions
- **Step Generation**: Creates natural language instructions from detected actions
- **Configurable Analysis**: Adjustable frame sampling and confidence thresholds
- **Multiple Output Formats**: Console output and JSON export

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. The tool will automatically download YOLOv8 model on first run.

## Usage

### Basic Usage
```bash
python video_to_instructions.py path/to/your/video.mp4
```

### Advanced Usage
```bash
python video_to_instructions.py input_video.mp4 \
    --output results.json \
    --frame-skip 10 \
    --confidence 0.7
```

### Parameters

- `video_path`: Path to the input video file (required)
- `--output, -o`: Save results to JSON file
- `--model, -m`: Path to custom model file
- `--frame-skip`: Analyze every Nth frame (default: 5)
- `--confidence`: Minimum confidence threshold (default: 0.5)

## Example Output

```
==================================================
STEP-BY-STEP INSTRUCTIONS
==================================================
Step 1: Person interacting with knife (at 0.0s)
Step 2: Engaging in cooking (at 0.0s)
Step 3: Person interacting with cutting board (at 2.0s)
Step 4: Person interacting with food (at 4.0s)

Total steps: 4
Video duration: 12.0 seconds
```

## Supported Video Formats

- MP4, AVI, MOV, MKV
- Any format supported by OpenCV

## Action Recognition

The tool recognizes common activities including:
- Cooking (knife, cutting board, pan, stove, food)
- Cleaning (cleaning supplies, sponge, broom, vacuum)
- Working (laptop, keyboard, mouse, desk, chair)
- Exercising (person, sports equipment, gym equipment)
- Repairing (tools, screwdriver, hammer, wrench)
- Organizing (box, container, shelf, storage)

## Technical Details

- Uses YOLOv8 for object detection (with OpenCV DNN fallback)
- Analyzes frames at configurable intervals
- Tracks object interactions and movements
- Generates natural language instructions based on detected patterns

## Troubleshooting

1. **CUDA not available**: The tool will automatically use CPU if CUDA is not available
2. **Model download issues**: Ensure internet connection for initial YOLOv8 model download
3. **Video format issues**: Try converting video to MP4 format if analysis fails
4. **Memory issues**: Increase frame-skip value to reduce memory usage

## Limitations

- Works best with clear, well-lit videos
- Limited to common household/office activities
- May miss subtle actions or rapid movements
- Performance depends on video quality and length