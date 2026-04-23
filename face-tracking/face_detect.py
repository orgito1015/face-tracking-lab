import csv
import os
import time
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Output locations (edit these if you want different specific files)
OUTPUT_DIR = "outputs"
CSV_PATH = os.path.join(OUTPUT_DIR, "face_tracking_log.csv")
VIDEO_PATH = os.path.join(OUTPUT_DIR, "face_tracking_dashboard.mp4")
PLOT_PATH = os.path.join(OUTPUT_DIR, "face_tracking_summary.png")


# Dashboard layout
FRAME_W, FRAME_H = 640, 480
PANEL_W = 420
DASHBOARD_W, DASHBOARD_H = FRAME_W + PANEL_W, FRAME_H


# Detection setup
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)


def detect_haar(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    return faces


def draw_series_panel(canvas, x, y, w, h, values, title, color, value_range):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (70, 70, 70), 1)
    cv2.putText(canvas, title, (x + 8, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    top_pad, bottom_pad, side_pad = 30, 12, 10
    plot_x0, plot_y0 = x + side_pad, y + top_pad
    plot_w, plot_h = w - 2 * side_pad, h - top_pad - bottom_pad

    cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x0 + plot_w, plot_y0 + plot_h), (40, 40, 40), 1)

    if len(values) < 2:
        return

    vmin, vmax = value_range
    if vmax <= vmin:
        vmax = vmin + 1

    pts = []
    total = len(values) - 1
    for i, val in enumerate(values):
        px = plot_x0 + int((i / total) * plot_w)
        clipped = float(np.clip(val, vmin, vmax))
        py = plot_y0 + int((1.0 - (clipped - vmin) / (vmax - vmin)) * plot_h)
        pts.append([px, py])

    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color, 2)
    cv2.putText(canvas, f"{values[-1]:.1f}", (x + w - 90, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


def save_summary_plot(times, yaw_values, cx_values, area_values):
    if len(times) < 2:
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(times, yaw_values, color="#01696F")
    axes[0].set_title("Yaw Error vs Time")
    axes[0].set_ylabel("Yaw")

    axes[1].plot(times, cx_values, color="#A84B2F")
    axes[1].set_title("Face X Position vs Time")
    axes[1].set_ylabel("X")

    axes[2].plot(times, area_values, color="#848456")
    axes[2].set_title("Face Area vs Time")
    axes[2].set_ylabel("Area")
    axes[2].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(VIDEO_PATH, fourcc, 20.0, (DASHBOARD_W, DASHBOARD_H))

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["time", "faces", "cx", "cy", "area", "yaw_error", "fps"])

        frame_count = 0
        start = time.time()

        times = []
        yaw_hist = []
        cx_hist = []
        area_hist = []

        yaw_live = deque(maxlen=220)
        cx_live = deque(maxlen=220)
        area_live = deque(maxlen=220)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            faces = detect_haar(frame)

            face_count = len(faces)
            cx, cy, area = 0, 0, 0

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 220, 120), 1)

            if face_count > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                cx, cy = x + w // 2, y + h // 2
                area = w * h
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

            frame_count += 1
            elapsed = max(time.time() - start, 1e-6)
            now = round(elapsed, 3)
            fps = frame_count / elapsed
            yaw_error = cx - FRAME_W // 2

            writer.writerow([now, face_count, cx, cy, area, yaw_error, round(fps, 2)])

            times.append(now)
            yaw_hist.append(yaw_error)
            cx_hist.append(cx)
            area_hist.append(area)

            yaw_live.append(yaw_error)
            cx_live.append(cx)
            area_live.append(area)

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {face_count}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(frame, "Press Q to quit", (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)

            dashboard = np.zeros((DASHBOARD_H, DASHBOARD_W, 3), dtype=np.uint8)
            dashboard[:, :FRAME_W] = frame

            panel_x = FRAME_W
            cv2.rectangle(dashboard, (panel_x, 0), (DASHBOARD_W - 1, DASHBOARD_H - 1), (55, 55, 55), 1)
            cv2.putText(dashboard, "Live Results", (panel_x + 12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            metrics_x = panel_x + 12
            metrics_y = 52
            line_h = 20
            metrics = [
                f"Time: {now:6.2f}s",
                f"FPS: {fps:5.1f}",
                f"Faces: {face_count}",
                f"Center X: {cx}",
                f"Center Y: {cy}",
                f"Area: {area}",
                f"Yaw error: {yaw_error}",
            ]
            for idx, text in enumerate(metrics):
                cv2.putText(
                    dashboard,
                    text,
                    (metrics_x, metrics_y + idx * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (180, 220, 255),
                    1,
                )

            chart_x = panel_x + 12
            chart_w = PANEL_W - 24
            chart_h = 126
            gap = 14
            first_y = 196

            draw_series_panel(
                dashboard,
                chart_x,
                first_y,
                chart_w,
                chart_h,
                list(yaw_live),
                "Yaw Error (cx - center)",
                (1, 105, 111),
                (-FRAME_W // 2, FRAME_W // 2),
            )
            draw_series_panel(
                dashboard,
                chart_x,
                first_y + chart_h + gap,
                chart_w,
                chart_h,
                list(cx_live),
                "Face X (pixels)",
                (47, 75, 168),
                (0, FRAME_W),
            )
            draw_series_panel(
                dashboard,
                chart_x,
                first_y + 2 * (chart_h + gap),
                chart_w,
                chart_h,
                list(area_live),
                "Face Area",
                (86, 132, 132),
                (0, FRAME_W * FRAME_H),
            )

            video_writer.write(dashboard)
            cv2.imshow("Face Tracking Dashboard", dashboard)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    save_summary_plot(times, yaw_hist, cx_hist, area_hist)
    print(f"Saved CSV:   {CSV_PATH}")
    print(f"Saved Video: {VIDEO_PATH}")
    print(f"Saved Plot:  {PLOT_PATH}")


if __name__ == "__main__":
    main()