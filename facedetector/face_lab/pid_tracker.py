import argparse
import csv
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2


@dataclass
class FaceInfo:
    cx: int
    cy: int
    area: int


class MockTello:
    def connect(self) -> None:
        print("MockTello connected")

    def takeoff(self) -> None:
        print("MockTello takeoff")

    def send_rc_control(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        print(f"RC -> lr={lr:>3}, fb={fb:>3}, ud={ud:>3}, yaw={yaw:>3}")

    def streamoff(self) -> None:
        print("MockTello stream off")


def clamp(value: float, low: int, high: int) -> int:
    return int(max(low, min(high, value)))


def find_face(frame, cascade: cv2.CascadeClassifier) -> Optional[FaceInfo]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    cx = int(x + w / 2)
    cy = int(y + h / 2)
    area = int(w * h)
    return FaceInfo(cx=cx, cy=cy, area=area)


def track(
    face: Optional[FaceInfo],
    frame_width: int,
    area_range: Tuple[int, int],
    kp: float,
    kd: float,
    prev_error: int,
) -> Tuple[int, int, int, int, int]:
    if face is None:
        return 0, 0, 0, 0, 0

    error_x = face.cx - frame_width // 2
    yaw = clamp(kp * error_x + kd * (error_x - prev_error), -100, 100)

    fb = 0
    if face.area > area_range[1]:
        fb = -20
    elif face.area < area_range[0] and face.area > 0:
        fb = 20

    lr = 0
    ud = 0
    return lr, fb, ud, yaw, error_x


def main() -> None:
    parser = argparse.ArgumentParser(description="Face tracking simulation with PID-like yaw control")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--duration-seconds", type=int, default=120)
    parser.add_argument("--output", default="rc_log.csv")
    parser.add_argument("--kp", type=float, default=0.4)
    parser.add_argument("--kd", type=float, default=0.1)
    parser.add_argument("--area-min", type=int, default=8000)
    parser.add_argument("--area-max", type=int, default=14000)
    args = parser.parse_args()

    tello = MockTello()
    tello.connect()
    tello.takeoff()

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam on index 0 or 1")

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("Could not load Haar cascade")

    area_range = (args.area_min, args.area_max)
    prev_error = 0
    start = time.time()

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "cx", "cy", "area", "error_x", "yaw", "fb", "lr", "ud"])

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                elapsed = time.time() - start
                if args.duration_seconds > 0 and elapsed >= args.duration_seconds:
                    print("Duration reached; stopping.")
                    break

                face = find_face(frame, cascade)
                frame_h, frame_w = frame.shape[:2]
                lr, fb, ud, yaw, error_x = track(face, frame_w, area_range, args.kp, args.kd, prev_error)
                prev_error = error_x

                tello.send_rc_control(lr, fb, ud, yaw)

                if face is not None:
                    cv2.circle(frame, (face.cx, face.cy), 4, (0, 255, 0), -1)
                    cv2.putText(frame, f"area={face.area}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"cx={face.cx}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    writer.writerow([f"{elapsed:.3f}", face.cx, face.cy, face.area, error_x, yaw, fb, lr, ud])
                else:
                    cv2.putText(frame, "No face", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    writer.writerow([f"{elapsed:.3f}", 0, 0, 0, 0, 0, 0, 0, 0])

                cv2.line(frame, (frame_w // 2, 0), (frame_w // 2, frame_h), (255, 255, 0), 1)
                cv2.putText(frame, f"yaw={yaw} fb={fb}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press Q to quit", (20, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
                cv2.imshow("pid_tracker", frame)

                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
        finally:
            tello.streamoff()
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
