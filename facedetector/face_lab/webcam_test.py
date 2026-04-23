import argparse
import platform
import time
from datetime import datetime

import cv2


def open_camera(preferred_index: int) -> cv2.VideoCapture:
    system = platform.system().lower()
    indexes = [preferred_index]
    if preferred_index != 1:
        indexes.append(1)

    for idx in indexes:
        if system == "windows":
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Camera opened on index {idx}")
            return cap
        cap.release()

    raise RuntimeError("Could not open webcam on index 0 or 1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Webcam test with FPS and timestamp overlay")
    parser.add_argument("--camera-index", type=int, default=0, help="Preferred webcam index")
    args = parser.parse_args()

    cap = open_camera(args.camera_index)

    prev_time = time.time()
    smoothed_fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed; stopping.")
                break

            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps = 1.0 / dt if dt > 0 else 0.0
            smoothed_fps = 0.9 * smoothed_fps + 0.1 * fps if smoothed_fps > 0 else fps

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, f"FPS: {smoothed_fps:5.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, timestamp, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press Q to quit", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("webcam_test", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
