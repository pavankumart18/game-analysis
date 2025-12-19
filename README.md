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
4. **Download Model Weights**
   - The system automatically downloads YOLO weights.
   - **SAM Weights**: Download `sam_vit_h_4b8939.pth` and place it in a `checkpoints/` folder.

## Usage

1. **Place Video**
   - Put your match footage in `videos/sample.mp4` (or update `VIDEO_PATH` in `main_dual.py`).

2. **Run Analysis**
   ```bash
   python main_dual.py
   ```

3. **View Results**
   - **`executive_summary_dual.txt`**: The final readable report.
   - **`node_insights_tactical.txt`**: Raw chronological tactical notes.
   - **`analysis_dual.json`**: Mathematical data dump (positions, metrics).

## Resume Capability
The system saves progress to `analysis_dual.json`. If you run the script again, it will skip the heavy video processing and instantly regenerate the LLM summary from the saved tracking data.
