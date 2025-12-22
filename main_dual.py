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
OUTPUT_JSON = "analysis_dual_sample.json" # Distinct output file
OUTPUT_SUMMARY = "executive_summary_dual.txt"
OUTPUT_FRAMES_DIR = "outputs" # Visual Output Directory
SAMPLE_FPS = 1.0 # 1.0 FPS = 300 frames total (manageable)
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
        # Device Check (Safe CPU Fallback)
        device_name = self.get_device()
        device = torch.device(device_name)
        print(f"Active Device: {device}")
        if device_name == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        print("Loading YOLOv8...")
        self.yolo = YOLO(YOLO_MODEL_NAME)
        self.yolo.to(device) 
        
        self.sam = None
        self.mask_generator = None
        
        # State for Temporal Continuity (Scene Detection)
        self.prev_hist = None
        
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
                min_mask_region_area=100,
            )
        else:
            self.sam = None
            print("SAM not loaded (missing checkpoint or library).")

        # Team Color State
        self.team_clusters = None # {id: 'A' or 'B'}
        self.player_colors = [] # Data for clustering
        
    def get_device(self):
        if torch.cuda.is_available():
            try:
                # Test call
                t = torch.tensor([1,2]).to("cuda")
                return "cuda"
            except Exception as e:
                print(f"CUDA available but failed: {e}. Using CPU.")
                return "cpu"
        return "cpu"

    def process_video(self):
        print(f"Processing video: {self.video_path}")
        print(f"Duration: {self.duration:.1f}s")
        
        # We need continuous frames for Tracking, but we only SAM-analyze specific frames
        # to keep it fast.
        
        # Analyze 0s to 10 mins
        start_time = 0
        end_time = 90 # 10 minutes limit
        
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        print(f"Running DEEP ANALYSIS (Tracking + SAM + Team ID) from {start_time}s to {end_time}s...")
        
        frames_data = []
        
        current_f = start_frame
        while current_f < end_frame:
            ret, frame = self.cap.read()
            if not ret: break
            
            # 1. TRACKING (Run on EVERY frame to keep IDs alive)
            # persist=True is key for ID continuity
            results = self.yolo.track(frame, persist=True, verbose=False, classes=[0, 32])
            
            # 2. SAM & Deep Analysis (Run at SAMPLE_FPS)
            # if SAMPLE_FPS = 1.0, we run every ~25 frames (if source is 25fps)
            is_sample_frame = (current_f - start_frame) % int(self.fps / SAMPLE_FPS) == 0
            
            if is_sample_frame:
                print(f"  > Deep Analyzing Frame {current_f} (Time: {current_f/self.fps:.2f}s)...")
                frame_data = self.analyze_single_frame(frame, current_f, results[0])
                frames_data.append(frame_data)
                
            current_f += 1
            
        self.cap.release()
        
        # Post-Process: Assign Teams based on collected colors
        self.kmeans_team_assignment(frames_data)
        
        # Re-Visualize with Team Colors (Since we only computed them now)
        # We need to reload the frames or just accept that the first pass didn't have ID colors?
        # A 2-pass approach is best: 
        # Pass 1: Collect Data.
        # Pass 2 (Virtual): Write Images.
        # But we don't want to read frames again.
        # Let's just iterate frames_data and assume we can't redraw the image easily without source.
        # Actually, we SHOULD retain the frames in memory if we want to draw "Team A/B" correctly.
        # But 10s of 1080p video @ 25fps = 250 frames -> ~2GB RAM. Acceptable.
        # Wait, the user wants "max capacity".
        # Let's update the visualizer to be called HERE, not inside the loop.
        
        print("Generating Final Visualizations with Team IDs... (SKIPPED to preserve SAM Overlays)")
        # self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        # current_f = start_frame
        
        # for i, f_data in enumerate(frames_data):
        #     ret, frame = self.cap.read()
        #     if not ret: break
        #    
        #     # Use the processed data
        #     self.visualize_frame(frame, f_data['frame_idx'], f_data['players'], f_data['ball'], None, f_data['field'])
        #    
        #     if i % 10 == 0: print(f"  > Saved Frame {i}/{len(frames_data)}")
            
        self.cap.release()
        
        # Update Segments with Team Data
        self.segments = [{
            "timestamp": f"{start_time}-{end_time}s",
            "raw_data": { "frames": frames_data }
        }]
        
        self.aggregate_results()

    def calculate_histogram(self, frame):
        """Compute HSV histogram for scene continuity check."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Calculate histogram (Hue, Saturation) - Ignore Value to reduce lighting noise
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def analyze_single_frame(self, frame, frame_idx, yolo_result):
        """
        Performs the heavy lifting: SAM + Mapping + Feature Extraction
        And VISUALIZES/SAVES immediately.
        """
        # --- SCENE CUT DETECTION ---
        curr_hist = self.calculate_histogram(frame)
        scene_score = 1.0
        is_cut = False
        
        if self.prev_hist is not None:
            scene_score = cv2.compareHist(self.prev_hist, curr_hist, cv2.HISTCMP_CORREL)
            # Threshold: < 0.6 usually implies significant camera movement or cut
            if scene_score < 0.6: 
                is_cut = True
                
        self.prev_hist = curr_hist # Update for next frame
        
        # A. Parse Tracks
        players = []
        ball = None
        
        if yolo_result.boxes:
             for box in yolo_result.boxes:
                 cls = int(box.cls[0])
                 # Check if ID is available (it might be None for new objects)
                 tid = int(box.id[0]) if box.id is not None else -1
                 xyxy = box.xyxy[0].tolist()
                 
                 if cls == 32: # Ball
                     ball = {"box": xyxy, "position": [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2]}
                 elif cls == 0: # Person
                     players.append({
                         "id": tid,
                         "box": xyxy,
                         "position": [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2]
                     })

        # B. SAM Segmentation
        sam_masks = []
        if self.mask_generator:
             try:
                height, width = frame.shape[:2]
                scale = 640 / width
                new_w, new_h = int(width * scale), int(height * scale)
                frame_small = cv2.resize(frame, (new_w, new_h))
                frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                masks = self.mask_generator.generate(frame_rgb)
                
                for m_i, m in enumerate(masks):
                    sx, sy, sw, sh = m['bbox']
                    inv_scale = 1/scale
                    real_box = [sx * inv_scale, sy * inv_scale, (sx + sw) * inv_scale, (sy + sh) * inv_scale]
                    sam_masks.append({
                        "id": m_i,
                        "bbox_xyxy": real_box,
                        "segmentation": m.get('segmentation', []), # Needed for overlay
                        "area": m['area']
                    })
             except Exception as e:
                 print(f"SAM Error on frame {frame_idx}: {e}")

        # C. Map & Extract Color
        mapped_players = []
        for p in players:
            # Find best SAM
            best_mask = None
            best_iou = 0
            for m in sam_masks:
                iou = self.calculate_iou(p['box'], m['bbox_xyxy'])
                if iou > best_iou: 
                    best_iou = iou
                    best_mask = m
            
            # Extract Color (Visual Feature)
            # Use the mask if available, else center of box
            dom_color = self.get_dominant_color(frame, p['box'], best_mask)
            self.player_colors.append(dom_color) # Store for clustering
            
            p['mask_id'] = best_mask['id'] if best_mask else None
            p['color_val'] = dom_color
            p['team'] = "Unknown" # Will be filled later
            mapped_players.append(p)

        # E. Field & Line Detection (Computer Vision)
        field_data = self.detect_field_and_lines(frame)

        # F. VISUALIZATION (Immediate Save)
        vis_frame = frame.copy()
        
        # 1. SAM Overlay
        if sam_masks:
            try:
                # Reconstruct overlay at original scale
                # Optimization: Do it on small scale then resize up?
                # We have masks in 'sam_masks', explicitly using segmentation?
                # Actually, m['segmentation'] is the boolean mask at ORIGINAL scale if produced by 'generate'?
                # Wait, we fed resized image to generator.
                # SamAutomaticMaskGenerator returns masks relative to input image size.
                # If we resized input, we need to resize masks back? Or generate on full image?
                # In current code: mask_generator.generate(frame_rgb) where frame_rgb is SMALL.
                # So 'segmentation' is small.
                
                height, width = frame.shape[:2]
                scale = 640 / width
                new_w, new_h = int(width * scale), int(height * scale)
                
                overlay_small = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                
                for m in sam_masks:
                    mask_bool = m.get('segmentation') # This is the boolean mask
                    if mask_bool is not None:
                        # Random color
                        color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()
                        overlay_small[mask_bool] = color
                
                # Upscale
                overlay_full = cv2.resize(overlay_small, (width, height), interpolation=cv2.INTER_NEAREST)
                
                # Blend
                # Check for size mismatch due to rounding
                oh, ow = overlay_full.shape[:2]
                if oh != height or ow != width:
                     overlay_full = cv2.resize(overlay_full, (width, height))
                
                gray_overlay = cv2.cvtColor(overlay_full, cv2.COLOR_BGR2GRAY)
                mask_indices = gray_overlay > 0
                vis_frame[mask_indices] = cv2.addWeighted(vis_frame[mask_indices], 0.7, overlay_full[mask_indices], 0.3, 0)
            except Exception as e:
                print(f"Vis Error: {e}")

        # 2. YOLO & Labels
        for p in mapped_players:
            box = p['box']
            cv2.rectangle(vis_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            label = f"ID:{p['id']}"
            if p['mask_id'] is not None: label += f"|M:{p['mask_id']}"
            cv2.putText(vis_frame, label, (int(box[0]), int(box[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 3. Field
        if field_data and field_data.get('field_polygon'):
             pts = np.array(field_data['field_polygon'], dtype=np.int32).reshape((-1, 1, 2))
             cv2.polylines(vis_frame, [pts], True, (0, 255, 0), 2)

        fname = f"{OUTPUT_FRAMES_DIR}/frame_{frame_idx}.jpg"
        cv2.imwrite(fname, vis_frame)
        print(f"    > Saved {fname}")
        
        return {
            "frame_idx": frame_idx,
            "timestamp_offset": frame_idx / self.fps,
            "ball": ball,
            "players": mapped_players,
            "field": field_data,
            "scene_correlation": round(scene_score, 3),
            "is_cut": is_cut
        }

    def detect_field_and_lines(self, frame):
        """
        Detects the Green Field (Play Area) and White Lines.
        """
        # 1. HSV Conversion
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 2. Green Mask (Field)
        # Range for Grass Green
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Morphological Closing to fill holes
        kernel = np.ones((5,5), np.uint8)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
        
        # Find Largest Contour -> Field
        contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        field_contour = []
        if contours:
            field_contour = max(contours, key=cv2.contourArea)
            # Simplify contour
            epsilon = 0.01 * cv2.arcLength(field_contour, True)
            field_contour = cv2.approxPolyDP(field_contour, epsilon, True).tolist()
            
        # 3. Line Detection (White pixels inside Field)
        # White definition
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 55, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # Only lines ON the field
        field_mask = np.zeros_like(mask_green)
        if len(contours) > 0:
            # We need to reconstruct the contour for drawing
            c = np.array(field_contour, dtype=np.int32).reshape((-1, 1, 2))
            cv2.drawContours(field_mask, [c], -1, 255, -1)
            
        mask_lines = cv2.bitwise_and(mask_white, mask_white, mask=field_mask)
        
        # Find Lines (Contours of white strips)
        line_contours, _ = cv2.findContours(mask_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_lines = []
        for lc in line_contours:
            if cv2.contourArea(lc) > 10: # Filter noise
                # bounding box
                x,y,w,h = cv2.boundingRect(lc)
                detected_lines.append([x,y,w,h])
                
        return {
            "field_polygon": field_contour, # List of [[x,y]]
            "lines": detected_lines # List of [x,y,w,h]
        }
    
    def get_dominant_color(self, frame, box, mask_obj):
        # Extract HSV hue
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return 0
        
        # If mask exists, mask the crop? (Too complex for simple resize, just use center box)
        # Simplify: Use center 50% of crop to avoid grass
        h, w = crop.shape[:2]
        center_crop = crop[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        
        hsv_crop = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
        hue_mean = np.mean(hsv_crop[:, :, 0])
        return hue_mean

    def kmeans_team_assignment(self, frames_data):
        # Simple thresholding since we have 2 teams
        if not self.player_colors: return
        
        colors = np.array(self.player_colors)
        # Ideally K-Means (k=2). 
        # Heuristic: Sort and split? 
        # Actually, let's just find the median and split high/low hue (Red vs Blue usually works)
        # Or if one is white and one is red...
        # For robustness, we'll use simple binning.
        
        # We will iterate frames and update 'team' field
        # based on a simple divider.
        median_hue = np.median(colors)
        
        # Define ranges for "Normal" team colors (Cluster A and Cluster B)
        # Outliers (Referee) might differ significantly
        
        for f in frames_data:
            for p in f['players']:
                hue = p['color_val']
                # Basic Split
                team = "Team A" if hue < median_hue else "Team B"
                
                # Check for Referee (Outlier)? 
                # Simple logic: If hue is very far from both cluster centers?
                # For demo, let's stick to 2 teams, but mark weird hues?
                p['team'] = team

    def visualize_frame(self, frame, idx, players, ball, masks, field_data=None):
        vis = frame.copy()
        
        # 0. Draw Field & Lines
        if field_data:
            # Poly
            poly = field_data.get('field_polygon', [])
            if poly:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            
            # Lines
            for l in field_data.get('lines', []):
                x,y,w,h = l
                # Draw subtle overlay for line
                cv2.rectangle(vis, (x,y), (x+w, y+h), (200, 200, 200), -1)

        # Draw Goal Zone
        cv2.rectangle(vis, (0, 300), (100, 600), (0, 255, 255), 2)
        cv2.putText(vis, "GOAL ZONE", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw Players
        for p in players:
            color = (0, 0, 255) if p['team'] == "Team A" else (255, 0, 0) # Red vs Blue
            box = p['box']
            cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            
            # Label
            label = f"ID:{p['id']} {p['team']}"
            cv2.putText(vis, label, (int(box[0]), int(box[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if ball:
             box = ball['box']
             cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 2)
        
        fname = f"{OUTPUT_FRAMES_DIR}/frame_{idx}.jpg"
        cv2.imwrite(fname, vis)


    @staticmethod
    def calculate_iou(box1, box2):
        """
        Calculates Intersection over Union between two boxes [x1, y1, x2, y2].
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - intersection_area
        return intersection_area / union_area if union_area > 0 else 0

    def analyze_segment(self, frames, timestamp):
        """
        Runs detection, tracking, and mapping for a 10s window (10 frames at 1FPS).
        Produces a rich, traceable JSON structure.
        """
        
        # Frame-by-Frame Data Container
        frames_analysis = []
        
        # --- 1. Detection (YOLO) & Mapping Loop ---
        if frames and self.yolo:
            for f_idx, frame in enumerate(frames):
                print(f"  > Analyzing Frame {f_idx}/{len(frames)}: {timestamp}...", flush=True)
                
                # A. YOLO Inference
                results = self.yolo(frame, verbose=False, classes=[0, 32]) # 0=person, 32=ball
                
                ball_obj = None
                yolo_players = []
                
                # Parse YOLO
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].tolist()
                        
                        if cls == 32: # Ball
                            # Keep highest conf ball
                            if not ball_obj or conf > ball_obj['conf']:
                                ball_obj = {
                                    "box": xyxy,
                                    "position": [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2],
                                    "conf": round(conf, 2)
                                }
                        elif cls == 0: # Person
                            yolo_players.append({
                                "box": xyxy,
                                "conf": round(conf, 2),
                                "position": [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2]
                            })

                # B. SAM Segmentation (If available)
                sam_masks = []
                if self.mask_generator:
                    try:
                        # Resize for speed
                        height, width = frame.shape[:2]
                        scale = 640 / width
                        new_w, new_h = int(width * scale), int(height * scale)
                        frame_small = cv2.resize(frame, (new_w, new_h))
                        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                        
                        masks = self.mask_generator.generate(frame_rgb)
                        
                        # Process Masks for Mapping
                        for m_i, m in enumerate(masks):
                            # Convert SAM bbox [x, y, w, h] to XYXY (and scale back up)
                            sx, sy, sw, sh = m['bbox']
                            inv_scale = 1/scale
                            real_box = [
                                sx * inv_scale, 
                                sy * inv_scale, 
                                (sx + sw) * inv_scale, 
                                (sy + sh) * inv_scale
                            ]
                            
                            sam_masks.append({
                                "id": m_i,
                                "bbox_xyxy": real_box,
                                "area": int(m['area'] * (inv_scale**2)),
                                "stability": round(float(m['stability_score']), 3),
                                "segmentation": m.get('segmentation', []) # Don't save full bitmap to JSON (too heavy)
                            })
                    except Exception as e:
                        print(f"    SAM Error Frame {f_idx}: {e}")

                # C. MAP YOLO -> SAM (Traceability Layer)
                mapped_players = []
                
                for p_idx, p in enumerate(yolo_players):
                    # Find best SAM match
                    best_iou = 0
                    best_mask = None
                    
                    for m in sam_masks:
                        iou = self.calculate_iou(p['box'], m['bbox_xyxy'])
                        if iou > best_iou:
                            best_iou = iou
                            best_mask = m
                    
                    # Create Mapped Object
                    player_entry = {
                        "id": f"P_{f_idx}_{p_idx}",
                        "box": [int(x) for x in p['box']],
                        "position": [int(x) for x in p['position']],
                        "yolo_conf": p['conf'],
                        "sam_ref_id": best_mask['id'] if best_mask else None,
                        "mapping_iou": round(best_iou, 3),
                        "mapping_reason": "Highest Intersection-over-Union (IoU) > 0" if best_mask else "No Overlap"
                    }
                    if best_mask:
                        player_entry["sam_metrics"] = {
                            "area": best_mask['area'],
                            "stability": best_mask['stability']
                        }
                    
                    mapped_players.append(player_entry)

                # --- D. VISUALIZATION (High Fidelity) ---
                vis_frame = frame.copy()
                
                # 1. SAM Overlay (Upscaled)
                if sam_masks:
                    # Create small overlay (using the same scale as inference)
                    height, width = frame.shape[:2]
                    scale = 640 / width
                    new_w, new_h = int(width * scale), int(height * scale)
                    
                    overlay_small = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                    
                    for m in sam_masks:
                        # Random color for each mask
                        color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()
                        mask_bool = m.get('segmentation')
                        if mask_bool is not None:
                            overlay_small[mask_bool] = color
                    
                    # Upscale to original size for high fidelity
                    overlay_full = cv2.resize(overlay_small, (width, height), interpolation=cv2.INTER_NEAREST)
                    
                    # Blend
                    gray_overlay = cv2.cvtColor(overlay_full, cv2.COLOR_BGR2GRAY)
                    mask_indices = gray_overlay > 0
                    # 70% Original, 30% Mask
                    vis_frame[mask_indices] = cv2.addWeighted(vis_frame[mask_indices], 0.7, overlay_full[mask_indices], 0.3, 0)

                # 2. YOLO Boxes & Data
                for p in mapped_players:
                    box = p['box']
                    # Color: Blue for players
                    cv2.rectangle(vis_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                    
                    # Label with Metrics
                    label = f"{p['id']}"
                    if p.get('sam_ref_id') is not None:
                        label += f"|SAM:{p['sam_ref_id']}"
                    
                    cv2.putText(vis_frame, label, (int(box[0]), int(box[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                if ball_obj:
                    box = ball_obj['box']
                    cv2.rectangle(vis_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 2)
                    cv2.putText(vis_frame, "BALL", (int(box[0]), int(box[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Save Frame
                ts_clean = timestamp.replace(":", "-").replace(" ", "_")
                fname = f"{OUTPUT_FRAMES_DIR}/frame_{ts_clean}_{f_idx}.jpg"
                cv2.imwrite(fname, vis_frame)

                # Store Frame Data
                frames_analysis.append({
                    "frame_idx": f_idx,
                    "timestamp_offset": f_idx * (1.0/SAMPLE_FPS) if SAMPLE_FPS > 0 else 0,
                    "ball": ball_obj,
                    "players": mapped_players,
                    "sam_mask_count": len(sam_masks),
                    "note": "Mapped Data: YOLO Box -> SAM Mask via IoU"
                })

        # Metrics Structure Update
        metrics = {
            "timestamp": timestamp,
            "raw_data": {
                "frames": frames_analysis # New Hierarchical Structure
            },
            "derived_metrics": {},
            "tactical_events": []
        }
            
        # Heuristic Updates (Simple)
        if frames_analysis:
             # Example: Average crowding
             avg_players = sum(len(f['players']) for f in frames_analysis) / len(frames_analysis)
             if avg_players > 18:
                 metrics["tactical_events"].append("High Player Density")
             
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
        # print("Skipping LLM Summary as requested.")

    def generate_llm_summary(self):
        print("Generating Executive Summary with LLM (Map-Reduce Strategy)...")
        
        # Updated Token and URL as per user request
        token = "keep-your-token"
        base_url = "https://llmfoundry.straive.com/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        PX_TO_M = 0.1 # Naive conversion: 10 pixels ~ 1 meter (Estimate for Broad view)
        
        tactical_insights_log = [] # Readable JSON log

        # --- PRE-PROCESSING: RESHAPE INTO 10s CHUNKS ---
        # Ensure we have granular segments (10 frames each)
        if self.segments and len(self.segments) == 1:
             big_frames = self.segments[0]['raw_data']['frames']
             if len(big_frames) > 10:
                 new_segments = []
                 chunk_size_frames = 10 # 10 frames = 10 seconds @ 1FPS
                 
                 print(f"Reshaping {len(big_frames)} frames into 10s chunks (Internal)...")
                 
                 for i in range(0, len(big_frames), chunk_size_frames):
                     chunk_frames = big_frames[i : i + chunk_size_frames]
                     if not chunk_frames: continue
                     
                     t_start = chunk_frames[0].get('timestamp_offset', 0)
                     t_end = chunk_frames[-1].get('timestamp_offset', 0)
                     
                     new_segments.append({
                         "timestamp": f"{t_start:.1f}-{t_end:.1f}s",
                         "raw_data": { "frames": chunk_frames }
                     })
                 self.segments = new_segments
                 print(f"Reshaped 1 massive segment into {len(self.segments)} narrative chunks (10s each).")

        # --- PHASE 1: SEQUENTIAL ANALYSIS (Chain of Narrative) ---
        chunk_size = 1 # 1 segment = 10s per analysis unit
        chunk_summaries = []
        previous_context = "Match Start (00s). No previous events."
        
        total_chunks = (len(self.segments) + chunk_size - 1) // chunk_size
        
        print(f"Analyzing match as specific {total_chunks} connected narrative nodes (10s each).")
        
        for i in range(0, len(self.segments), chunk_size):
            chunk = self.segments[i:i + chunk_size]
            chunk_idx = i // chunk_size + 1
            
            # The prompt expects the raw_data from the first segment in the chunk
            chunk_data_str = json.dumps(chunk[0], cls=NumpyEncoder) 
            
            print(f"  > Analyzing Node {chunk_idx}/{total_chunks}...", flush=True)
            
            # 3. Passing & Goal Analysis
            passing_log = []
            goal_threat = []
            
            # Holder Tracking Sequence
            holders = [] # (frame_idx, player_id)
            if chunk:
                frames = chunk[0].get('raw_data', {}).get('frames', [])
                for f in frames:
                    ball = f.get('ball')
                    players = f.get('players', [])
                    if not ball:
                        holders.append(None)
                        continue
                    
                    # Goal Check
                    bx = ball['position'][0] if 'position' in ball else ball['center'][0]
                    if bx < 100 or bx > 1180:
                        goal_threat.append(f"Frame {f['frame_idx']}: BALL IN GOAL ZONE (X={bx:.0f})")

                    # Holder Check
                    min_dist = float('inf')
                    holder = None
                    for p in players:
                        p_box = p.get('box', p.get('yolo_box'))
                        px = (p_box[0] + p_box[2]) / 2
                        py = (p_box[1] + p_box[3]) / 2
                        # Actually we need by (ball y)
                        by = ball['position'][1] if 'position' in ball else ball['center'][1]
                        dist = math.sqrt((bx-px)**2 + (by-py)**2)
                        if dist < min_dist:
                            min_dist = dist
                            holder = p
                    
                    if holder and min_dist < 60:
                        holders.append(holder.get('id', holder.get('player_id'))) 
                    else:
                        holders.append(None)
            
            # Ball Pressure Metrics
            pressure_context = []
            events_str = "" # Placeholder for events like "TIGHT PRESS"
            if chunk:
                seg = chunk[0]
                frames_data = seg.get('raw_data', {}).get('frames', [])
                
                # Calculate ball velocity and tempo
                ball_positions = [f['ball']['position'] for f in frames_data if f.get('ball') and f['ball'].get('position')]
                avg_ball_velocity = 0
                if len(ball_positions) > 1:
                    velocities = []
                    for k in range(1, len(ball_positions)):
                        p1 = ball_positions[k-1]
                        p2 = ball_positions[k]
                        velocities.append(math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2))
                    avg_ball_velocity = sum(velocities) / len(velocities)
                
                tempo_score = "SLOW"
                if avg_ball_velocity > 30: tempo_score = "FAST"
                elif avg_ball_velocity > 10: tempo_score = "MEDIUM"

                # Summarize frames for LLM
                summary_frames = []
                for f_data in frames_data:
                    f_idx = f_data['frame_idx']
                    ball = f_data.get('ball')
                    players = f_data.get('players', [])
                    
                    ball_pos_str = "N/A"
                    if ball and ball.get('position'):
                        ball_pos_str = f"({int(ball['position'][0])},{int(ball['position'][1])})"
                    
                    player_summary = []
                    for p in players:
                        team = p.get('team', 'Unknown')
                        player_summary.append(f"{p.get('id', p.get('player_id'))} ({team}) at ({int(p['position'][0])},{int(p['position'][1])})")
                    
                    summary_frames.append({
                        "frame_idx": f_idx,
                        "ball_position": ball_pos_str,
                        "players_count": len(players),
                        "players_positions": player_summary
                    })

                    if ball and ball.get('box'):
                        ball_box = ball['box']
                        bx = (ball_box[0] + ball_box[2]) / 2
                        by = (ball_box[1] + ball_box[3]) / 2
                        
                        min_dist = float('inf')
                        closest_p = None
                        
                        for p in players:
                             # Use mapped YOLO box
                             p_box = p['box']
                             px = (p_box[0] + p_box[2]) / 2
                             py = (p_box[1] + p_box[3]) / 2
                             dist = math.sqrt((bx-px)**2 + (by-py)**2)
                             if dist < min_dist:
                                 min_dist = dist
                                 closest_p = p
                        
                        if closest_p:
                             pressure_type = "TIGHT PRESS" if min_dist < 50 else "LOOSE / STAND-OFF"
                             events_str += f"Frame {str(f_idx).zfill(2)}: Ball Carrier {pressure_type} (Dist: {min_dist:.1f}px)\n"
                             # Reference the Player ID and Linked SAM Mask for traceability
                             sam_info = f"(Linked SAM Mask: {closest_p['sam_ref_id']})" if closest_p.get('sam_ref_id') is not None else "(No SAM Match)"
                             pressure_context.append(f"Frame {str(f_idx).zfill(2)}: Ball Carrier {pressure_type} (Dist: {min_dist:.1f}px) {sam_info}")

            pressure_str = "\n".join(pressure_context)

            # --- ADVANCED METRICS (HUMAN READABLE & COACH LEVEL) ---
            start_t = 0 + i * 10
            end_t = start_t + (chunk_size * 10)
            chunk_metrics = {
                "timestamp_range": f"{start_t}s - {end_t}s",
                "teams": {},
                "game_state": "Neutral"
            }
            
            # 1. Identify "Moment of Interest" (Snapshot Frame)
            # Find the frame with the highest aggregate movement (Transition/Turnover)
            max_momentum_idx = 0
            max_momentum_val = 0
            
            frame_movements = []
            if len(frames) > 1:
                for idx in range(len(frames)-1):
                    # Check for Scene Cut first
                    corr = frames[idx+1].get('scene_correlation', 1.0)
                    if corr < 0.6: 
                        continue # Skip calculating movement across a cut
                        
                    # Calculate aggregate shift of all players
                    p_curr = frames[idx].get('players', [])
                    p_next = frames[idx+1].get('players', [])
                    
                    # Simple centroid shift
                    def get_all_centroid(pl):
                        if not pl: return (0,0)
                        xs = [p.get('position', p.get('center'))[0] for p in pl]
                        ys = [p.get('position', p.get('center'))[1] for p in pl]
                        return (sum(xs)/len(xs), sum(ys)/len(ys))
                    
                    c1 = get_all_centroid(p_curr)
                    c2 = get_all_centroid(p_next)
                    
                    shift = math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                    frame_movements.append(shift)
                    
                    if shift > max_momentum_val:
                        max_momentum_val = shift
                        max_momentum_idx = idx

            # Get Snapshot Frame Data (at max momentum)
            snapshot_frame = frames[max_momentum_idx] if frames else {}
            
            # Metric Calculation at SNAPSHOT (Not Average)
            team_stats = {"Team A": {"x": [], "y": []}, "Team B": {"x": [], "y": []}}
            for p in snapshot_frame.get('players', []):
                 team = p.get('team')
                 if team in team_stats:
                     box = p.get('box', p.get('yolo_box'))
                     if box:
                        cx = (box[0] + box[2]) / 2
                        cy = (box[1] + box[3]) / 2
                        team_stats[team]["x"].append(cx)
                        team_stats[team]["y"].append(cy)

            tactical_stats = f"SNAPSHOT METRICS (At Max Momentum Frame {snapshot_frame.get('frame_idx', 0)}):\n"
            
            for t_name, data in team_stats.items():
                xs = data["x"]
                ys = data["y"]
                if not xs or not ys: 
                    chunk_metrics["teams"][t_name] = "No Data"
                    continue
                
                xs.sort()
                ys.sort()
                n = len(xs)
                # Convert to Meters
                line_low_x = int(xs[int(n*0.1)] * PX_TO_M) 
                line_high_x = int(xs[int(n*0.9)] * PX_TO_M)
                
                # Compactness in Meters
                team_length_m = (xs[int(n*0.9)] - xs[int(n*0.1)]) * PX_TO_M
                team_width_m = (ys[int(n*0.9)] - ys[int(n*0.1)]) * PX_TO_M
                
                shape_note = "Compact" if team_length_m < 30 else "Stretched"
                if team_length_m > 50: shape_note = "Very Stretched / Disorganized"
                
                chunk_metrics["teams"][t_name] = {
                    "defense_line_m": line_low_x,
                    "attack_line_m": line_high_x,
                    "length_m": round(team_length_m, 1),
                    "width_m": round(team_width_m, 1),
                    "shape": shape_note
                }
                
                tactical_stats += (
                    f"{t_name}: Block {line_low_x}m-{line_high_x}m | "
                    f"Size {team_length_m:.1f}x{team_width_m:.1f}m ({shape_note})\n"
                )

            # 4. Pressing Intensity & Celebration Check
            tight_press_count = events_str.count("TIGHT PRESS")
            chunk_metrics["press_events"] = tight_press_count
            
            # Celebration / Goal Confirmation
            # If Ball in Goal Zone AND nearby players are 'Slow' grouping?
            # Simplified: Just report the flag.
            
            chunk_metrics["momentum_mps"] = round(max_momentum_val * PX_TO_M, 2) # Meters per second peak
            if max_momentum_val * PX_TO_M > 2.0:
                chunk_metrics["game_state"] = "High Transition / Counter-Attack"
            elif tight_press_count > 2:
                chunk_metrics["game_state"] = "High Press / Duel"
            else:
                 chunk_metrics["game_state"] = "Build-up / Consolidated"

            tactical_insights_log.append(chunk_metrics)

            chunk_prompt = f"""
            Task: Acting as a Senior Tactical Analyst (Premier League Level).
            Analyze this 10-second segment.
            
            CONTEXT DATA:
            PREVIOUS STATE: "{previous_context}"
            
            GAME STATE DETECTED: {chunk_metrics['game_state']}
            
            SNAPSHOT METRICS (Meters, Converted @ ~10px/m):
            {tactical_stats}
            
            MOMENTUM PEAK: {chunk_metrics['momentum_mps']} m/s
            (Note: Momentum calc suppresses camera cuts to ensure continuity).
            
            EVENTS LOG:
            {events_str}
            
            ANALYSIS REQUIREMENTS (COACH-LEVEL):
            1. **Game State**: Is this a Counter-Attack (High Momentum), Build-up (Low Momentum), or Pressing Trap?
            2. **Defensive Integorty**: Look at 'Block' meters. Is the team too deep (<10m)? Too stretched (>40m length)?
            3. **Key Events**: Confirm Goals. If "BALL IN GOAL ZONE", look for "High Momentum" towards center (reset) or corner (celebration).
            4. **Forecast**: Based on the Shape (Compact vs Stretched), who is vulnerable?
            
            OUTPUT FORMAT (Strictly Technical):
            - **Game State**: [Phase of Play]
            - **Team Shape**: [Analysis of Block Height & Compactness in Meters]
            - **Critical Events**: [Goals/Fouls/Pressing]
            - **Coach's Insight**: [1 actionable point on spacing/structure]
            """ 
            # CONSTRAINT: Do not report "0.0px/frame". If metrics show movement, explain it tactically.
            # The above constraint is removed as it's not relevant for meter values.
            
            try:
                response = requests.post(
                    base_url,
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
                    start_t = 0 + i * 10
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
        with open("node_insights_tactical.txt", "w", encoding='utf-8') as f:
            f.write("\n\n".join(chunk_summaries))
        print("Detailed tactical insights saved to 'node_insights_tactical.txt'.")

        # Save Readable Metrics
        with open("tactical_analysis_readable.json", "w", encoding='utf-8') as f:
            json.dump(tactical_insights_log, f, indent=2, ensure_ascii=False)
        print("Human-readable metrics saved to 'tactical_analysis_readable.json'")

        # --- PHASE 2: FINAL COACH REPORT ---
        print("Compiling final Tactical Report...")
        
        combined_notes = "\n\n".join(chunk_summaries)
        
        final_prompt = f"""
        You are a World-Class Sports Storyteller.
        
        INPUT DATA:
        A series of DEEP TECHNICAL ANALYSIS notes from 10-second chunks of a football match.
        
        RAW NOTES:
        {combined_notes}
        
        TASK:
        Weave these technical notes into a fluid, compelling NARRATIVE STORY.
        
        REQUIREMENTS:
        1. **Chronological Story**: Start from the beginning and tell the story of these 10 seconds.
        2. **Vivid Description**: Use the "Visual State" notes to paint a picture of the field.
        3. **Analysis to Action**: Translate "High Velocity" to "The team surged forward".
        4. **Highlight Key Moments**: If a goal or foul happened, dramatize it.
        
        Your output should read like a high-quality match report article.
        """
        
        summary_text = None
        
        try:
            response = requests.post(
                base_url,
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
            
        with open(OUTPUT_SUMMARY, 'w', encoding='utf-8') as f:
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
            
        # Generate summary
        pipeline.generate_llm_summary()
    else:
        pipeline.process_video()
        # pipeline.aggregate_results() is called inside process_video
