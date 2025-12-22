# AI Football Tactical Analyst (YOLO + SAM + LLM)

## Overview
This project implements an autonomous tactical analysis pipeline for football (soccer) video. It combines state-of-the-art Computer Vision (YOLOv8, SAM) with generative AI (GPT-4o) to produce professional, coach-ready match reports.

The system bypasses traditional statistics to focus on *tactical behavior*: shape, compactness, pressing triggers, and transition patterns.

## Features
- **Dual-Model Vision Upgrade**: 
  - **YOLOv8**: Detects players and the ball with high frequency.
  - **SAM (Segment Anything)**: Analyzes spatial control and formation density.
- **Physics-Based Metrics**:
  - **Ball Pressure**: Calculates real-time proximity of defenders to the ball carrier.
  - **Compactness**: Measures squad dispersion (X/Y spread) dynamically.
  - **Tempo**: Tracks ball velocity to determine play speed.
- **Narrative Intelligence**:
  - Uses a **Map-Reduce** LLLM strategy to analyze 10s chunks before synthesizing a full match story.
  - **Output 1**: Executive Summary (Actionable "Do This -> Because This" format).
  - **Output 2**: Narrative Insight Story (Malcolm Gladwell style, ~4 mins read).

## Methodology & Metrics

The system derives tactical insights using Physics-based calculations on Computer Vision outputs:

### 1. Ball Pressure (Pressing Intensity)
- **Definition**: Measures how tightly the ball carrier is being marked.
- **Calculation**: Euclidean distance between the Ball centroid and the nearest Player centroid.
- **Formula**: `√((x2-x1)² + (y2-y1)²)` < 50 pixels triggers "TIGHT PRESS".
- **Insight**: Distinguishes between aggressive pressing and standoff defense.

### 2. Squad Compactness (Shape)
- **Definition**: Measures the discipline of the team shape (Lateral & Vertical).
- **Calculation**: Standard Deviation (`stdev`) of all player centroids in the current frame.
- **Formula**: 
  - `Compactness_X = stdev([p.x for p in players])` (Width)
  - `Compactness_Y = stdev([p.y for p in players])` (Depth)
- **Insight**: High dispersion indicates broken lines or fatigue; low dispersion indicates a disciplined block.

### 3. Game Tempo (Ball Momentum)
- **Definition**: The velocity of play.
- **Calculation**: Delta of ball position between sampled frames (approx. every 0.3s).
- **Thresholds**: 
  - **> 100px/interval**: High Tempo (Counter-attacks, long balls).
  - **< 30px/interval**: Low Tempo (Static structure).

### 4. Possession Logic
- **Holder Detection**: A player is deemed the "Ball Carrier" if they are the nearest player to the ball and within **60 pixels**.

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA (Recommended for SAM)

### Steps
1. **Clone the Repository**
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Setup Environment**
   Create a `.env` file in the root directory:
   ```env
   LLMFOUNDRY_TOKEN=your_api_token_here
   ```
### Model Setup

#### 1. Segment Anything Model (SAM) - Critical
This project uses the specific **ViT-H** (Huge) checkpoint (~2.5 GB) for maximum accuracy.

1. Create a `checkpoints` directory in the project root.
2. Download the model file and place it inside.

**Command Line (PowerShell/Bash):**
```bash
mkdir checkpoints
curl -o checkpoints/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
**Direct Download Link**: [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

#### 2. YOLOv8
The `ultralytics` library will **automatically download** the `yolov8n.pt` (Nano) model on the first run. No manual action is required.

## Usage

1. **Place Video**
   - **Source**: [Tactical Cam - Man City vs Inter](https://youtu.be/PrG_9JO38Ow?si=Rl0ou40DFPf3voUs)
   - Download the video, rename it to `sample.mp4`, and place it in the `videos/` directory.
   - (Alternatively, update `VIDEO_PATH` in `main_dual.py` to point to your file).

2. **Run Analysis**
   ```bash
   python main_dual.py
   ```

3. **View Results**
   - **`executive_summary_dual.txt`**: The final readable report.
   - **`node_insights_tactical.txt`**: Raw chronological tactical notes.
   - **`analysis_dual.json`**: Mathematical data dump (positions, metrics).

## Resume Capability
The system saves progress to `analysis_dual_sample.json` (or the configured JSON name). If you run the script again, it will skip the heavy video processing and instantly regenerate the LLM summary from the saved tracking data.

## Output Files Explained

### 1. `analysis_dual_sample.json` (The Core Data)
This is the raw, frame-by-frame mathematical dump of the computer vision pipeline. It serves as the "Source of Truth" for all downstream analysis.
**Structure:**
- **`metadata`**: Video props (FPS, resolution, duration).
- **`segments`**: List of 10-second video chunks.
  - **`raw_data` -> `frames`**:
    - **`ball`**: Position `(x, y)` and Bounding Box.
    - **`players`**: List of detected players with:
       - **`id`**: Unique Tracking ID.
       - **`team`**: Team assignment (Team A / Team B) based on color clustering.
       - **`position`**: Centroid `(x, y)`.
       - **`box`**: Bounding Box.
    - **`scene_correlation`**: Score (0.0 - 1.0) indicating visual continuity (detects camera cuts).

### 2. `tactical_analysis_readable.json` (The Metrics Layer)
A processed, human-readable summary of the match metrics, aggregated into 10-second "Narrative Nodes". This bridges the gap between raw pixels and tactical concepts.
**Structure:** (List of Objects)
- **`timestamp_range`**: e.g., "0s - 10s".
- **`game_state`**: Automatically detected phase (e.g., "High Transition", "Build-up").
- **`teams`**:
  - **`defense_line_m`**: Position of the last defender (approx. meters).
  - **`length_m`**: Vertical distance between front and back lines (Compactness).
  - **`shape`**: Descriptive label (e.g., "Compact", "Stretched", "Disorganized").
- **`momentum_mps`**: aggregate "meters per second" of the squad movement (Momentum).
- **`press_events`**: Count of "High Press" trigger events (Defenders < 2m from Ball).

### 3. `node_insights_tactical.txt` (The Chain of Thought)
This text file contains the raw, chronological "Thought Process" of the AI Analyst. It consists of 30 separate analysis notes (one for each 10s chunk).
- **Usage**: Use this to debug *why* the AI concluded something.
- **Content**: visualizes the field, identifies specific press events, and logs the AI's immediate reaction to that specific 10s clip.

### 4. `executive_summary_dual.txt` (The Final Report)
The "Coach-Level" output. This is a synthesized document created by a Map-Reduce LLM strategy that reads all 30 `node_insights` and summaries the broader narrative.
**Content:**
- **Match Narrative**: 3-4 paragraphs telling the story of the game.
- **Tactical Breakdown**: Specific section on Defensive Structure and Transition Speed.
- **Coach Recommendations**: Bullet points on what the losing team needs to fix (e.g., "Defensive Line is too deep").
