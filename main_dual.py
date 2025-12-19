import cv2
import numpy as np
import json
import os
import sys
import math
from collections import defaultdict
from pathlib import Path

# Try importing dependencies
try:
    from ultralytics import YOLO
    import torch
    import requests # Added requests import
except ImportError:
    print("Missing dependencies. Please run: pip install ultralytics torch torchvision opencv-python numpy requests")
    sys.exit(1)

# Optional SAM import (will warn if missing)
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    print("Warning: segment_anything not installed. SAM features will be simulated/limited.")
    SAM_AVAILABLE = False

# ==========================================
# CONFIGURATION
# ==========================================
VIDEO_PATH = os.path.join("videos", "sample.mp4")
OUTPUT_JSON = "analysis_dual.json" # Distinct output file
OUTPUT_SUMMARY = "executive_summary_dual.txt"
SAMPLE_FPS = 0.3 # 0.3 FPS = ~3 frames per 10s segment
SEGMENT_DURATION = 10  # seconds
YOLO_MODEL_NAME = "yolov8n.pt" # Nano model for speed
SAM_CHECKPOINT = os.path.join("checkpoints", "sam_vit_h_4b8939.pth")
SAM_MODEL_TYPE = "vit_h"

# ==========================================
# UTILS
# ==========================================
def ensure_sam_weights():
    """Helper to download SAM weights if not present (simplified for demo)"""
    if not SAM_AVAILABLE: return
    if not os.path.exists(SAM_CHECKPOINT):
        print(f"Downloading SAM weights {SAM_CHECKPOINT}...")
        # This is a placeholder. Real code would use requests/wget to download key models.
        # For this exercise, we assume the user might have them or we warn.
        # But to be 'runnable', if we can't download, we might need a fallback logic.
        print("Please download SAM weights manually to enable full segmentation.")

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class FootballAnalyticsPipeline:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video Resolution: {width}x{height}")
        self.duration = self.total_frames / self.fps
        
        # Models
        # STRICT GPU ENFORCEMENT
        device = torch.device("cuda")
        print(f"Active Device: {device} (Torch v{torch.__version__})")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        print("Loading YOLOv8...")
        self.yolo = YOLO(YOLO_MODEL_NAME)
        self.yolo.to(device) 
        
        self.sam = None
        self.mask_generator = None
        if SAM_AVAILABLE and os.path.exists(SAM_CHECKPOINT):
            print("Loading SAM (Automatic Mask Generator)...")
            self.sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
            self.sam.to(device=device)
            
            # Use Automatic Generator to find EVERYTHING (replaces YOLO)
            # OPTIMIZATION: points_per_side=12 (144 pts) vs Default 32 (1024 pts)
            # This makes it ~7x faster while keeping reasonable density.
            print("Initializing SAM with reduced grid (12x12)...")
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=12,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,  # Requires open-cv to run post-processing
            )
        else:
            self.sam = None
            print("SAM not loaded (missing checkpoint or library).")

        # State
        self.segments = [] # List of {"timestamp": str, ...}
        
    def process_video(self):
        print(f"Processing video: {self.video_path}")
        print(f"Duration: {self.duration:.1f}s")
        # Ensure we pick up at least 3 frames. 
        # FPS/SAMPLE_FPS gives skip step. if sample_fps < 1, skip step > FPS.
        frames_to_skip = int(self.fps / SAMPLE_FPS)
        
        # Start from 40s (Segment 4) to 300s (Segment 30, i.e., 5 mins)
        start_segment = 4
        end_segment = 30 
        
        print(f"Skipping first {start_segment*10}s. Analyzing from 40s to 5mins with DUAL MODEL (YOLO+SAM)...")
        
        # We need to seek the video to 40s first
        start_frame_global = int(start_segment * SEGMENT_DURATION * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_global)
        
        frame_idx = start_frame_global
        
        for i in range(start_segment, end_segment):
            if i * SEGMENT_DURATION >= self.duration: break
            
            start_time = i * SEGMENT_DURATION
            end_time = (i + 1) * SEGMENT_DURATION
            timestamp_str = f"{start_time}-{end_time}s"
            
            print(f"Analyzing Segment {i+1}/{end_segment}: {timestamp_str} (3 Frames Dual-Analysis)", flush=True)
            
            # Extract frames for this segment
            segment_frames = []
            segment_frame_indices = []
            
            # GPU Mode: Process frame-by-frame (Sampled)
            start_f_seg = int(start_time * self.fps)
            end_f_seg = int(end_time * self.fps)
            
            current_f = start_f_seg
            # Ensure cap is at correct position (redundant if read is sequential but safe)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_f)
            
            while current_f < end_f_seg:
                ret, frame = self.cap.read()
                if not ret: break
                
                # Sample at SAMPLE_FPS
                if (current_f - start_f_seg) % frames_to_skip == 0:
                    segment_frames.append(frame)
                    segment_frame_indices.append(current_f)
                
                current_f += 1
            
            # Analyze Segment
            metrics = self.analyze_segment(segment_frames, timestamp_str)
            self.segments.append(metrics)
            
        self.cap.release()
        self.aggregate_results()

    def analyze_segment(self, frames, timestamp):
        """
        Runs detection, tracking, and metric computation for a 10s window.
        Returns strict JSON schema structure.
        """
        
        # Aggregators for this segment
        player_detections = [] # list of (frame_idx, box, team, id)
        ball_positions = []
        
        # --- 1. Detection (YOLO) ---
        # ENABLED NOW
        for f_idx, frame in enumerate(frames):
            # Run YOLO
            results = self.yolo(frame, verbose=False, classes=[0, 32]) # 0=person, 32=ball
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    
                    if cls == 32: # Ball
                        ball_positions.append(xyxy)
                    elif cls == 0: # Person
                        player_detections.append({
                            "frame_rel_idx": f_idx, # Relative frame index in this batch of 3
                            "box": xyxy,
                            "conf": round(conf, 2)
                        })

        # --- 2. Team Clustering (Simple Color Heuristic) ---
        # Pick the middle frame to determine teams
        if frames:
            mid_frame = frames[len(frames)//2]
            teams_map = self.assign_teams(mid_frame, [p['box'] for p in player_detections if p['frame_rel_idx'] == len(frames)//2])
        else:
            teams_map = {}

        # --- 3. SAM Segmentation (Automatic - Raw Outputs) ---
        sam_raw_dump = []
        
        if self.mask_generator and frames:
            print(f"Running SAM Automatic Generator on {len(frames)} frames...")
            
            for f_idx, frame in enumerate(frames):
                try:
                    # OPTIMIZATION: Resize for speed (ViT-H is too slow on full HD)
                    height, width = frame.shape[:2]
                    scale = 640 / width
                    new_w, new_h = int(width * scale), int(height * scale)
                    frame_small = cv2.resize(frame, (new_w, new_h))
                    
                    print(f"  > Frame {f_idx}: Processing at {new_w}x{new_h}...", flush=True)
                    frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                    
                    # Raw Output from SAM
                    masks = self.mask_generator.generate(frame_rgb)
                    print(f"    SAM found {len(masks)} masks.")
                    
                    for m in masks:
                        sam_raw_dump.append({
                            "frame_rel_idx": f_idx, # Track which frame this belongs to
                            "bbox": [int(x) for x in m['bbox']], # [x, y, w, h]
                            "area": int(m['area']),
                            "predicted_iou": round(float(m['predicted_iou']), 3),
                            "stability_score": round(float(m['stability_score']), 3),
                            "crop_box": [int(x) for x in m['crop_box']] if 'crop_box' in m else [],
                            "point_coords": m.get('point_coords', []) 
                        })
                        
                except Exception as e:
                    print(f"SAM Inference error on frame {f_idx}: {e}")
                    pass
                
        # Metrics Structure Update
        metrics = {
            "timestamp": timestamp,
            "raw_data": {
                "sam_objects": sam_raw_dump,   # THE RAW SAM OUTPUT (List of all segments)
                "yolo_objects": player_detections, # NEW: Add YOLO data
                "ball_positions": ball_positions
            },
            "derived_metrics": {},
            "tactical_events": []
        }
            
        # Add basic event detection based on SAM Data
        mask_count = len(masks) if self.mask_generator and frames else 0
        
        # Heuristic 1: Complexity/Crowding
        if mask_count > 60:
            metrics["tactical_events"].append("Crowded Area / High Activity")
        elif mask_count < 20:
            metrics["tactical_events"].append("Open Play / Transition")
            
        # Heuristic 2: Camera Proximity (Average Area of objects)
        if sam_raw_dump:
            avg_area = np.mean([m['area'] for m in sam_raw_dump])
            if avg_area > 5000:
                metrics["tactical_events"].append("Close-up / Replay")
            elif avg_area < 500:
                metrics["tactical_events"].append("Wide Tactical View")

        return metrics

    def assign_teams(self, frame, boxes):
        # Placeholder for color clustering
        # In real impl, crop players -> K-Means (k=2) on jersey colors
        return {}
    
    def aggregate_results(self):
        # 1. Write JSON
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(self.segments, f, indent=2, cls=NumpyEncoder)
        print(f"Analytics saved to {OUTPUT_JSON}")
        
        # 2. Generate LLM Summary
        self.generate_llm_summary()

    def generate_llm_summary(self):
        print("Generating Executive Summary with LLM (Map-Reduce Strategy)...")
        
        from dotenv import load_dotenv
        load_dotenv()
        token = os.getenv("LLMFOUNDRY_TOKEN")
        
        if not token:
            print("WARNING: LLMFOUNDRY_TOKEN not found in .env. LLM calls will fail.")
            token = "MISSING_TOKEN"

        headers = {
            "Authorization": f"Bearer {token}:my-test-project",
            "Content-Type": "application/json"
        }
        # --- PHASE 1: SEQUENTIAL ANALYSIS (Chain of Narrative) ---
        chunk_size = 2 # 2 segments * 10s = 20s per analysis unit
        chunk_summaries = []
        previous_context = "Match Start (40s mark). Teams settling into shape."
        
        total_chunks = (len(self.segments) + chunk_size - 1) // chunk_size
        
        print(f"Analyzing match as specific {total_chunks} connected narrative nodes (10s each).")
        
        for i in range(0, len(self.segments), chunk_size):
            chunk = self.segments[i:i + chunk_size]
            chunk_idx = i // chunk_size + 1
            
            # Serialize just this chunk
            # The prompt expects the raw_data from the first segment in the chunk
            chunk_data_str = json.dumps(chunk[0], cls=NumpyEncoder) 
            
            print(f"  > Analyzing Node {chunk_idx}/{total_chunks}...", flush=True)
            
            # Prepare Ball Pressure Metrics
            pressure_context = []
            if chunk:
                # ... (Logic remains same, just renaming output string)
                # ... (Copy existing proximity logic)
                seg = chunk[0]
                raw = seg.get('raw_data', {})
                balls = raw.get('ball_positions', [])
                players = raw.get('yolo_objects', [])
                
                players_by_frame = defaultdict(list)
                for p in players:
                    players_by_frame[p['frame_rel_idx']].append(p)
                
                for f_idx, p_list in players_by_frame.items():
                    if f_idx < len(balls): 
                        ball_box = balls[f_idx]
                        bx = (ball_box[0] + ball_box[2]) / 2
                        by = (ball_box[1] + ball_box[3]) / 2
                        
                        min_dist = float('inf')
                        closest_p = None
                        
                        for p in p_list:
                             p_box = p['box']
                             px = (p_box[0] + p_box[2]) / 2
                             py = (p_box[1] + p_box[3]) / 2
                             dist = math.sqrt((bx-px)**2 + (by-py)**2)
                             if dist < min_dist:
                                 min_dist = dist
                                 closest_p = p
                        
                        if closest_p:
                             # Inference: < 50px is "Tight Pressure", > 100px is "Loose / Space"
                             pressure_type = "TIGHT PRESS" if min_dist < 50 else "LOOSE / STAND-OFF"
                             pressure_context.append(f"Frame {f_idx}: Ball Carrier under {pressure_type} (Dist: {min_dist:.1f}px) at {closest_p['box']}")
            
            pressure_str = "\n".join(pressure_context)

            # --- ADVANCED METRICS CALCULATION ---
            # 1. Squad Dispersion (Compactness)
            # Collect all player centroids in this chunk
            all_px, all_py = [], []
            if chunk:
                for seg in chunk:
                    players = seg.get('raw_data', {}).get('yolo_objects', [])
                    for p in players:
                         box = p['box']
                         all_px.append((box[0] + box[2]) / 2)
                         all_py.append((box[1] + box[3]) / 2)
            
            compactness_x = 0
            compactness_y = 0
            if len(all_px) > 1:
                import statistics
                compactness_x = statistics.stdev(all_px)
                compactness_y = statistics.stdev(all_py)
            
            dispersion_score = f"X-Spread: {compactness_x:.1f}px (Width), Y-Spread: {compactness_y:.1f}px (Depth)"
            
            # 2. Ball Momentum (Tempo)
            ball_dists = []
            if chunk:
                # Naive: Distance between consequent balls
                balls = chunk[0].get('raw_data', {}).get('ball_positions', [])
                if len(balls) > 1:
                    for b_i in range(len(balls) - 1):
                        b1 = balls[b_i]
                        b2 = balls[b_i+1]
                        c1 = ((b1[0]+b1[2])/2, (b1[1]+b1[3])/2)
                        c2 = ((b2[0]+b2[2])/2, (b2[1]+b2[3])/2)
                        dist = math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                        ball_dists.append(dist)
            
            avg_ball_velocity = sum(ball_dists)/len(ball_dists) if ball_dists else 0
            tempo_score = "HIGH" if avg_ball_velocity > 100 else "MODERATE" if avg_ball_velocity > 30 else "SLOW"

            # 3. Passing & Goal Analysis (Heuristic)
            passing_log = []
            goal_threat = []
            
            # Re-run holder identification for sequence tracking
            holders = [] # [(frame_idx, box)]
            if chunk:
                seg = chunk[0]
                raw = seg.get('raw_data', {})
                balls = raw.get('ball_positions', [])
                players = raw.get('yolo_objects', [])
                
                players_by_frame = defaultdict(list)
                for p in players:
                    players_by_frame[p['frame_rel_idx']].append(p)
                
                for f_idx in range(len(balls)):
                    ball_box = balls[f_idx]
                    bx = (ball_box[0] + ball_box[2]) / 2
                    by = (ball_box[1] + ball_box[3]) / 2
                    
                    # Check Goal Zone (Assuming 720p/1080p roughly, <100 or >1180)
                    if bx < 100 or bx > 1180:
                        goal_threat.append(f"Frame {f_idx}: BALL IN GOAL ZONE (X={bx:.0f})")
                    
                    # Find Holder
                    min_dist = float('inf')
                    holder = None
                    if f_idx in players_by_frame:
                        for p in players_by_frame[f_idx]:
                             p_box = p['box']
                             px = (p_box[0] + p_box[2]) / 2
                             py = (p_box[1] + p_box[3]) / 2
                             dist = math.sqrt((bx-px)**2 + (by-py)**2)
                             if dist < min_dist:
                                 min_dist = dist
                                 holder = p
                    
                    if holder and min_dist < 60: # Valid holder
                        holders.append(holder['box'])
                    else:
                        holders.append(None)
            
            # Detect Transfers
            for h_test in range(len(holders)-1):
                h1 = holders[h_test]
                h2 = holders[h_test+1]
                if h1 and h2 and h1 != h2:
                     passing_log.append(f"PASS DETECTED: {h1} -> {h2}")
            
            events_str = "\n".join(passing_log + goal_threat)
            if not events_str: events_str = "No clear passing sequences or goal events detected."

            chunk_prompt = f"""
            You are an Elite Tactical Analyst (UEFA Pro License Level).
            
            PREVIOUS CONTEXT: "{previous_context}"
            
            CURRENT DATA MAP (20s Segment):
            1. PLAYER POSITIONS (YOLO): {json.dumps(chunk[0]['raw_data'].get('yolo_objects', []) if 'raw_data' in chunk[0] else [], cls=NumpyEncoder)}
            2. SPATIAL DENSITY (SAM): {json.dumps(chunk[0]['raw_data'].get('sam_objects', []) if 'raw_data' in chunk[0] else [], cls=NumpyEncoder)}
            3. BALL PRESSURE METRICS:
               {pressure_str}
            4. ADVANCED SQUAD METRICS:
               - COMPACTNESS (Dispersion): {dispersion_score}
               - TEMPO (Ball Velocity): {avg_ball_velocity:.1f}px/frame ({tempo_score})
            5. GAME EVENTS (PASSING & GOALS):
               {events_str}
            
            TASK:
            Diagnose the TACTICAL STATE.
            
            ANALYZE:
            1. SHAPE: Dispersion analysis.
            2. PASSING CHAINS: Describe how players are connecting (see 'PASS DETECTED').
            3. GOAL EVENTS: Did a goal occur? (Check 'BALL IN GOAL ZONE').
            4. VULNERABILITY: Space opening up.
            
            OUTPUT:
            Strictly 5 bullet points using these headers:
            - **Shape**: [Observation]
            - **Passing Network**: [Who is passing to whom?]
            - **Goal/Event**: [Goal scored? Shot taken?]
            - **Vulnerability**: [Observation]
            - **Key Player Action**: [Who is impacting flow?]
            """
            
            try:
                response = requests.post(
                    "https://llmfoundry.straive.com/openai/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": "You are a UEFA Pro License Analyst."},
                            {"role": "user", "content": chunk_prompt}
                        ]
                    }
                )
                if response.status_code == 200:
                    result_text = response.json()['choices'][0]['message']['content']
                    # Timestamp relative to 40s start
                    # i is the segment index relative to the loaded list
                    # Each segment is 10s.
                    # start_time = 40 + i * 10
                    start_t = 40 + i * 10
                    end_t = start_t + (chunk_size * 10)
                    chunk_summaries.append(f"[{start_t}s - {end_t}s]:\n{result_text}")
                    # Update context for next loop so it "flows"
                    previous_context = result_text 
                else:
                    chunk_summaries.append(f"[Node {chunk_idx}]: [Analysis Failed]")
                    print(f"Node {chunk_idx} failed: {response.text}")
                    
            except Exception as e:
                print(f"Error analyzing node {chunk_idx}: {e}")
        
        # Save Raw Insights to File
        with open("node_insights_tactical.txt", "w") as f:
            f.write("\n\n".join(chunk_summaries))
        print("Detailed tactical insights saved to 'node_insights_tactical.txt'.")

        # --- PHASE 2: FINAL COACH REPORT ---
        print("Compiling final Tactical Report...")
        
        combined_notes = "\n\n".join(chunk_summaries)
        
        final_prompt = f"""
        You are a World-Class Football Writer (like Malcolm Gladwell meets Jonathan Wilson).
        
        DATA SOURCE:
        Detailed technical match breakdown (attached below).
        
        RAW ANALYST NOTES:
        {combined_notes}
        
        TASK:
        Synthesize these notes into a FINAL REPORT.
        
        STRICT FORMAT REQUIREMENTS:
        
        ## PART 1: EXECUTIVE SUMMARY (FOR COACHES)
        - Provide 2-4 Actionable Bullets.
        - Format: "**Action** -> Because **Rationale**".
        - Example: "**Increase midfield compactness** because spacing expands by 30% after possession loss."
        - NO TECHNICAL JARGON (No YOLO, SAM, AI, JSON).
        
        ## PART 2: INSIGHT STORY
        - Style: Narrative, compelling, "Malcolm Gladwell" style. (Target length: 800-1000 words, ~4 mins read).
        - Title: "The Hidden Rhythm: Velocity and Compactness"
        - Structure:
          1. **Opening Hook**: Start with an observation about the game's deceptive simplicity vs hidden complexity.
          2. **What We Observed**: Describe the specific patterns found in the data (spacing, pressure triggers, movement, passing chains).
          3. **The Key Patterns**: Dedicate paragraphs to 3 key findings (e.g., Transition Speed, Pressing Gaps, Fatigue).
          4. **Why This Matters**: Explain the coaching impact of these patterns.
          5. **Future Outlook**: Brief closing on using this observation method.
        
        CONSTRAINT:
        - Everything must be grounded in the provided RAW NOTES.
        - Do NOT mention "The AI detected" or "The model found". Say "The footage shows" or "We observed".
        """
        
        summary_text = None
        
        try:
            response = requests.post(
                "https://llmfoundry.straive.com/openai/v1/chat/completions",
                headers=headers,
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are Pep Guardiola's Assistant Coach."},
                        {"role": "user", "content": final_prompt}
                    ]
                }
            )
            
            if response.status_code == 200:
                summary_text = response.json()['choices'][0]['message']['content']
                print("Final Executive Summary generated successfully.")
            else:
                print(f"Final LLM failed: {response.text}")
                
        except Exception as e:
            print(f"Final LLM error: {e}")
        
        # Fallback
        if not summary_text:
            summary_text = "Analysis failed or API error. \n\nRaw Chunk Notes:\n" + combined_notes
            
        with open(OUTPUT_SUMMARY, 'w') as f:
            f.write(summary_text)
        print(f"Executive Summary saved to {OUTPUT_SUMMARY}")



if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found at {VIDEO_PATH}")
    # Run Pipeline
    pipeline = FootballAnalyticsPipeline(VIDEO_PATH)
    
    # OPTIONAL: Resume from existing JSON if available (User Request)
    if os.path.exists(OUTPUT_JSON):
        print(f"Found existing {OUTPUT_JSON}. Loading data without re-processing video...")
        with open(OUTPUT_JSON, 'r') as f:
            pipeline.segments = json.load(f)
        # Directly generate summary
        pipeline.generate_llm_summary()
    else:
        pipeline.process_video()
        # pipeline.aggregate_results() is called inside process_video

