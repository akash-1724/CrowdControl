#!/usr/bin/env python3
# counts_summary.py
# Usage: python counts_summary.py counts_per_frame.csv
# Produces counts_smoothed.csv, counts_per_sec.csv, counts_report.txt

import sys, csv, math
from statistics import mean

def moving_average(arr, window):
    if window <= 1:
        return arr[:]
    out = []
    half = window//2
    n = len(arr)
    for i in range(n):
        lo = max(0, i-half)
        hi = min(n, i+half+1)
        out.append(mean(arr[lo:hi]))
    return out

def main():
    if len(sys.argv) < 2:
        print("Usage: python counts_summary.py counts_per_frame.csv")
        return
    infile = sys.argv[1]
    frames = []
    counts = []
    # Expect columns: frame, people (or people_in_frame)
    with open(infile, newline='') as f:
        r = csv.DictReader(f)
        # find which column likely holds people
        cols = r.fieldnames
        people_col = None
        for c in ['people','people_in_frame','people_entered','count']:
            if c in cols:
                people_col = c; break
        if people_col is None:
            # fallback: assume second column
            people_col = cols[1]
        for row in r:
            frames.append(int(row['frame']))
            counts.append(int(float(row[people_col])))

    # smoothing (frame-level) - window depends on fps; assume fps from counts: try to infer fps by frames/sec later
    # default window: 3 frames
    smoothed = moving_average(counts, window=3)

    # compute fps estimate from contiguous frames (if frames contiguous)
    if len(frames) >= 2:
        diffs = [frames[i+1]-frames[i] for i in range(len(frames)-1)]
        median_diff = sorted(diffs)[len(diffs)//2]
        if median_diff > 0:
            fps_est = round(1.0/median_diff, 3) if median_diff != 0 else None
        else:
            fps_est = None
    else:
        fps_est = None

    # safer: ask for fps? but we will compute per-second by grouping frames into second bins using fps= (assume 24 if not found)
    fps = 24.0
    # attempt to read FPS from counts file name or environment can't be done here, so default to 24
    # If your detector prints FPS, you can pass it; otherwise adjust fps below if you know it.
    # Aggregate counts per-second using simple frame -> second mapping
    sec_counts = {}
    for f, c in zip(frames, counts):
        second = int(f // int(fps))
        sec_counts.setdefault(second, 0)
        sec_counts[second] = max(sec_counts[second], c)  # use max people visible in that sec (better for headcount)

    # Total estimate: best is the last 'people_entered' column if exists else unique estimator (next script)
    total_est = None
    # if input has people_entered column use that
    if 'people_entered' in cols:
        # re-read quickly
        with open(infile, newline='') as f:
            r = csv.DictReader(f)
            last = None
            for row in r:
                last = row
            if last and 'people_entered' in last:
                total_est = int(float(last['people_entered']))

    # fallback total estimated as max visible (not perfect)
    if total_est is None:
        total_est = max(counts) if counts else 0

    # write smoothed CSV
    out_sm = infile.replace('.csv', '_smoothed.csv')
    with open(out_sm, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'people_raw', 'people_smoothed'])
        for fr, raw, s in zip(frames, counts, smoothed):
            w.writerow([fr, raw, round(s,3)])

    # write per-second CSV
    out_sec = infile.replace('.csv', '_persec.csv')
    with open(out_sec, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['second','people_peak_in_second'])
        for sec in sorted(sec_counts.keys()):
            w.writerow([sec, sec_counts[sec]])

    # peak info
    peak_frame = frames[counts.index(max(counts))] if counts else None
    peak_val = max(counts) if counts else 0
    avg_visible = round(sum(counts)/len(counts), 3) if counts else 0

    # write simple report
    out_report = infile.replace('.csv', '_report.txt')
    with open(out_report, 'w') as f:
        f.write(f"Counts summary for: {infile}\n")
        f.write(f"Frames processed: {len(frames)}\n")
        f.write(f"FPS used for per-second aggregation (assumed): {fps}\n")
        f.write(f"Total estimated (best-effort): {total_est}\n")
        f.write(f"Peak visible in a frame: {peak_val} at frame {peak_frame}\n")
        f.write(f"Average visible (per frame): {avg_visible}\n")
        f.write("\nFiles written:\n")
        f.write(f"- smoothed: {out_sm}\n")
        f.write(f"- per-second: {out_sec}\n")
        f.write(f"- report: {out_report}\n")

    print("Wrote:", out_sm, out_sec, out_report)
    print("Estimated total:", total_est)

if __name__ == '__main__':
    main()
