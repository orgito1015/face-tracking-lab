"""
Session 1 — Webcam test: live FPS overlay + HH:MM:SS timestamp.
No argparse — beginners run this directly: python webcam_test.py
"""
import sys
import time
from datetime import datetime

import cv2


def open_camera() -> cv2.VideoCapture:
    indexes = [0, 1]
    for idx in indexes:
        if sys.platform == "win32":
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"[INFO] Webcam opened on index {idx}.")
            return cap
        cap.release()
    raise RuntimeError("Could not open webcam on index 0 or 1.")


def main() -> None:
    cap = open_camera()

    frame_count = 0
    start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Frame read failed; stopping.")
                break

            frame_count += 1
            elapsed = max(time.time() - start, 1e-6)
            fps = frame_count / elapsed

            timestamp = datetime.now().strftime("%H:%M:%S")

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),      # green
                2,
            )
            cv2.putText(
                frame,
                timestamp,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 215, 255),    # gold
                2,
            )

            cv2.imshow("Webcam Feed", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
