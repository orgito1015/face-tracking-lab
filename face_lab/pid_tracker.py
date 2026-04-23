"""
Session 2 — MockTello face-tracking simulator with PID loop and CSV logging.

Usage:
    python pid_tracker.py --camera-index 0 --output rc_log.csv --kp 0.4 --kd 0.1
"""
import argparse
import csv
import sys
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Module-level constants (informational — not mutated at runtime)
# ---------------------------------------------------------------------------
MAX_SPEED = 30

FRAME_W, FRAME_H = 640, 480
FACE_TIMEOUT = 3.0


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    left_label: str = "",
    right_label: str = "",
) -> np.ndarray:
    """Return a single frame with *left* and *right* placed side-by-side."""
    h = max(left.shape[0], right.shape[0])

    def _fit(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == h:
            return img.copy()
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h))

    left = _fit(left)
    right = _fit(right)

    for img, label in ((left, left_label), (right, right_label)):
        if label:
            cv2.putText(
                img, label, (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA,
            )
            cv2.putText(
                img, label, (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA,
            )

    divider = np.full((h, 3, 3), 80, dtype=np.uint8)
    return np.hstack([left, divider, right])


# ---------------------------------------------------------------------------
# MockTello — simulated drone with integrated webcam and CSV logger
# ---------------------------------------------------------------------------

class MockTello:
    """Simulated Tello drone. Uses the local webcam and logs RC commands to CSV."""

    class _FR:
        """Frame-reader shim (mirrors DJITelloPy's BackgroundFrameRead)."""

        def __init__(self, cap: cv2.VideoCapture) -> None:
            self._cap = cap

        @property
        def frame(self) -> np.ndarray:
            ok, f = self._cap.read()
            return f if ok else np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

    def __init__(self, log_file: str = "rc_log.csv", camera_index: int = 0) -> None:
        # Open webcam (CAP_DSHOW on Windows for lower latency)
        if sys.platform == "win32":
            self._cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(camera_index)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(1)

        if not self._cap.isOpened():
            raise RuntimeError("Could not open webcam on index 0 or 1.")

        # Open CSV log
        self._log = open(log_file, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._log)
        self._writer.writerow(["time", "lr", "fb", "ud", "yaw", "cx", "cy", "area"])

        self._start = time.time()

    def connect(self) -> None:
        print("[SIM] Connected.")

    def get_battery(self) -> int:
        return 100

    def streamon(self) -> None:
        pass  # webcam is already open in __init__

    def streamoff(self) -> None:
        self._cap.release()
        self._log.close()

    def takeoff(self) -> None:
        print("[SIM] Takeoff.")

    def land(self) -> None:
        print("[SIM] Land.")

    def send_rc_control(
        self, lr: int, fb: int, ud: int, yaw: int,
        cx: int = 0, cy: int = 0, area: int = 0
    ) -> None:
        t = round(time.time() - self._start, 3)
        self._writer.writerow([t, lr, fb, ud, yaw, cx, cy, area])
        print(f"[t={t:5.2f}s] LR:{lr:+4d} FB:{fb:+4d} UD:{ud:+4d} YAW:{yaw:+4d}")

    def get_frame_read(self) -> "_FR":
        return MockTello._FR(self._cap)


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------

def find_face(frame: np.ndarray, cascade: cv2.CascadeClassifier):
    """Return (annotated_frame, cx, cy, area). cx/cy/area are 0 when no face found."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))

    if len(faces) == 0:
        return frame, 0, 0, 0

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    cx = x + w // 2
    cy = y + h // 2
    area = w * h
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame, cx, cy, area


# ---------------------------------------------------------------------------
# PID tracking
# ---------------------------------------------------------------------------

def track(cx: int, cy: int, area: int, prev_err: int, kp: float, kd: float,
          dt: float = 0.033):
    """Compute RC commands from face position. Returns (lr, fb, ud, yaw, err)."""
    err = cx - FRAME_W // 2
    yaw = int(np.clip(
        kp * err + kd * (err - prev_err) / dt,
        -MAX_SPEED, MAX_SPEED
    ))

    ud_err = cy - FRAME_H // 2
    ud = int(np.clip(-kp * ud_err, -MAX_SPEED, MAX_SPEED)) if abs(ud_err) > 40 else 0

    fb = -15 if area > 14000 else (15 if 0 < area < 8000 else 0)

    return 0, fb, ud, yaw, err


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Face tracking simulator with PID control and CSV logging"
    )
    parser.add_argument("--camera-index", type=int, default=0,
                        help="Webcam index (default: 0)")
    parser.add_argument("--output", default="rc_log.csv",
                        help="CSV output path (default: rc_log.csv)")
    parser.add_argument("--kp", type=float, default=0.4,
                        help="PID proportional gain (default: 0.4)")
    parser.add_argument("--kd", type=float, default=0.1,
                        help="PID derivative gain (default: 0.1)")
    args = parser.parse_args()

    # Override module-level PID constants from CLI
    kp = args.kp
    kd = args.kd

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        raise RuntimeError("Could not load haarcascade_frontalface_default.xml.")

    drone = MockTello(log_file=args.output, camera_index=args.camera_index)
    drone.connect()
    drone.streamon()
    drone.takeoff()

    reader = drone.get_frame_read()
    prev_err = 0
    last_face = time.time()

    try:
        while True:
            frame = cv2.resize(reader.frame, (FRAME_W, FRAME_H))
            raw = frame.copy()
            frame, cx, cy, area = find_face(frame, cascade)

            if area > 0:
                last_face = time.time()
            elif time.time() - last_face > FACE_TIMEOUT:
                drone.send_rc_control(0, 0, 0, 0)
                cv2.putText(
                    frame, "NO FACE - HOVERING", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )
                cv2.imshow("Tracker", _side_by_side(raw, frame, "Camera", "Tracking"))
                cv2.waitKey(1)
                continue

            lr, fb, ud, yaw, prev_err = track(cx, cy, area, prev_err, kp, kd)
            drone.send_rc_control(lr, fb, ud, yaw, cx, cy, area)

            cv2.imshow("Tracker", _side_by_side(raw, frame, "Camera", "Tracking"))
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        drone.land()
        drone.streamoff()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
