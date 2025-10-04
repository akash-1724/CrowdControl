#!/usr/bin/env python3
# annotate_counts.py
# Usage: python annotate_counts.py input_video.mp4 counts_per_frame.csv output_video.mp4
# Draws a clean per-frame overlay: Frame N, People: X

import sys
import csv
import cv2

def load_counts(csvfile):
    """Loads frame counts from the specified CSV file."""
    by_frame = {}
    with open(csvfile, newline='') as f:
        reader = csv.DictReader(f)
        # Auto-detect the 'people' count column
        people_col = 'people'
        if people_col not in reader.fieldnames:
            people_col = reader.fieldnames[1] # Fallback to the second column

        for row in reader:
            by_frame[int(row['frame'])] = int(row[people_col])
    return by_frame

def main():
    if len(sys.argv) < 4:
        print("Usage: python annotate_counts.py <input_video> <counts_csv> <output_video>")
        return

    video_in_path = sys.argv[1]
    counts_csv_path = sys.argv[2]
    video_out_path = sys.argv[3]

    counts = load_counts(counts_csv_path)

    cap = cv2.VideoCapture(video_in_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_in_path}")
        return

    # Get video properties for the output file
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        people_count = counts.get(frame_idx, 0)

        # Draw a clean, black rectangle for the text background
        cv2.rectangle(frame, (5, 5), (250, 65), (0, 0, 0), -1)
        # Add Frame number text
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        # Add People count text
        cv2.putText(frame, f"People: {people_count}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ Success! Created verification video: {video_out_path}")

if __name__ == "__main__":
    main()