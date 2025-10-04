#!/usr/bin/env python3
"""
detect_count_flow.py
Simple YOLOv8-based person detector that:
- reads a video
- detects people per frame
- saves counts_per_frame.csv (people visible per frame) and person_flow.csv
- calculates a true total number of entrants by tracking line crossing
- writes annotated output video (output.mp4 by default) with total count
- saves the final total count to total_count.txt

Usage:
    python detect_count_flow.py input_video.mp4 output_video.mp4 counts.csv flow.csv
Example:
    python detect_count_flow.py test_video.mp4 output.mp4 counts_per_frame.csv person_flow.csv
"""

import sys
import time
import csv
import cv2
import pandas as pd
from ultralytics import YOLO

def safe_box_vals(box):
    """
    Extract box coordinates, class and confidence robustly across ultralytics versions.
    Returns: (x1,y1,x2,y2,cls,conf)
    """
    try:
        xyxy = box.xyxy.cpu().numpy()[0]
        x1, y1, x2, y2 = map(int, xyxy.tolist())
        cls = int(box.cls.cpu().numpy()[0])
        conf = float(box.conf.cpu().numpy()[0])
        return x1, y1, x2, y2, cls, conf
    except Exception:
        pass
    try:
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy.tolist())
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        return x1, y1, x2, y2, cls, conf
    except Exception:
        return 0, 0, 0, 0, -1, 0.0

def main(inp_vid, out_vid, counts_csv, flow_csv, conf_thresh=0.35, imgsz=640):
    t0 = time.time()
    print("[*] Loading YOLOv8 model (yolov8n)...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(inp_vid)
    if not cap.isOpened():
        print(f"[!] ERROR: cannot open input video: {inp_vid}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[*] Video opened: {inp_vid} ({w}x{h} @ {fps:.1f} fps)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_vid, fourcc, fps, (w, h))

    # --- Tactical Addition for True Total Count ---
    entry_line_y = h * 0.8  # An imaginary line at 80% of the video height
    tracked_ids = set()
    total_entrants = 0
    # ------------------------------------------

    frame_idx = 0
    counts_rows = []
    flow_rows = []

    print("[*] Starting processing... Press Ctrl+C to stop early.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, imgsz=imgsz, conf=conf_thresh, verbose=False)
            r = results[0]

            person_count = 0
            detections = []
            boxes = getattr(r, "boxes", None)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2, cls, conf = safe_box_vals(box)
                    if cls != 0:
                        continue
                    person_count += 1
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    detections.append((x1, y1, x2, y2, cx, cy, conf))

            # --- MODIFIED: draw boxes and track entry ---
            current_frame_ids = set()
            for i, (x1,y1,x2,y2,cx,cy,conf) in enumerate(detections):
                # Use a combination of centroid and box size for a more stable temp ID
                temp_id = f"{round(cx / 10)}_{round(cy / 10)}_{round((x2-x1)/10)}"
                current_frame_ids.add(temp_id)

                # Check if the centroid has crossed the entry line and is a new track
                if cy > entry_line_y and temp_id not in tracked_ids:
                    total_entrants += 1
                    tracked_ids.add(temp_id) # Mark this ID as counted

                # --- Original drawing code (unchanged) ---
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,0), 2)
                cv2.circle(frame, (cx,cy), 3, (0,200,0), -1)
                cv2.putText(frame, f"{conf:.2f}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            
            # Prune old IDs that are no longer in the scene to avoid memory issues on long videos
            if frame_idx % 100 == 0:
                tracked_ids.intersection_update(current_frame_ids)

            # --- MODIFIED: Add Total Entrants to overlay ---
            cv2.rectangle(frame, (5,5), (250,85), (0,0,0), -1) # Made the box bigger
            cv2.putText(frame, f"Frame: {frame_idx}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"People Now: {person_count}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Total Entrants: {total_entrants}", (10,75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2) # Added total

            out.write(frame)

            counts_rows.append({"frame": frame_idx, "people": person_count})
            for (x1, y1, x2, y2, cx, cy, conf) in detections:
                flow_rows.append({"frame": frame_idx, "cx": cx, "cy": cy, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf})

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user")
    finally:
        cap.release()
        out.release()

    pd.DataFrame(counts_rows).to_csv(counts_csv, index=False)
    pd.DataFrame(flow_rows).to_csv(flow_csv, index=False)

    # --- Tactical Addition: Save Total Count ---
    with open("total_count.txt", "w") as f:
        f.write(str(total_entrants))
    # ------------------------------------------

    dt = time.time() - t0
    print(f"[*] Done. Processed {frame_idx} frames in {dt:.1f}s ({frame_idx/dt:.1f} fps)")
    print(f"[*] Final Total Entrants: {total_entrants}")
    print(f"[*] Outputs: {out_vid}, {counts_csv}, {flow_csv}, total_count.txt")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python detect_count_flow.py input_video.mp4 output_video.mp4 counts.csv flow.csv")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])