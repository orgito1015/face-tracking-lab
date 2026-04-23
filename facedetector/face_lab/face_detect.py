import argparse
from pathlib import Path
import time
from typing import List, Tuple
import urllib.request

import cv2
import mediapipe as mp


Box = Tuple[int, int, int, int]


def ensure_mediapipe_face_model() -> str:
    model_dir = Path(__file__).resolve().parent / "models"
    model_path = model_dir / "blaze_face_short_range.tflite"
    if model_path.exists():
        return str(model_path)

    model_dir.mkdir(parents=True, exist_ok=True)
    url = (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
    )
    try:
        print("Downloading MediaPipe face model (first run only)...")
        urllib.request.urlretrieve(url, model_path)
        return str(model_path)
    except Exception as exc:
        raise RuntimeError(
            "Could not download MediaPipe model. Check internet access or manually place "
            "blaze_face_short_range.tflite in face_lab/models/."
        ) from exc


def get_mediapipe_detector_kind() -> str:
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
        return "legacy"
    if hasattr(mp, "tasks"):
        return "tasks"
    raise RuntimeError(
        "MediaPipe face detection API is unavailable in this environment. "
        "Reinstall with: pip install --upgrade mediapipe"
    )


def create_mediapipe_detector():
    kind = get_mediapipe_detector_kind()
    if kind == "legacy":
        detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        return kind, detector

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = ensure_mediapipe_face_model()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )
    detector = vision.FaceDetector.create_from_options(options)
    return kind, detector


def detect_haar(frame, cascade: cv2.CascadeClassifier) -> List[Box]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def detect_mediapipe(frame, detector, detector_kind: str) -> List[Box]:
    if detector_kind == "legacy":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)
        if not result.detections:
            return []

        h, w = frame.shape[:2]
        boxes: List[Box] = []
        for det in result.detections:
            rel = det.location_data.relative_bounding_box
            x = max(0, int(rel.xmin * w))
            y = max(0, int(rel.ymin * h))
            bw = int(rel.width * w)
            bh = int(rel.height * h)
            boxes.append((x, y, bw, bh))
        return boxes

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    if not result.detections:
        return []

    boxes: List[Box] = []
    for det in result.detections:
        bbox = det.bounding_box
        boxes.append((int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)))
    return boxes


def draw_boxes(frame, boxes: List[Box], color=(0, 255, 0)) -> None:
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def run_haar(camera_index: int, benchmark_seconds: int) -> float:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam for Haar backend")

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("Could not load Haar cascade")

    start = time.time()
    frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            boxes = detect_haar(frame, cascade)
            draw_boxes(frame, boxes, color=(0, 255, 255))

            frames += 1
            elapsed = max(1e-6, time.time() - start)
            fps = frames / elapsed
            cv2.putText(frame, f"Haar FPS: {fps:5.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, "Press Q to quit", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
            cv2.imshow("face_detect - Haar", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
            if benchmark_seconds > 0 and elapsed >= benchmark_seconds:
                break

        avg_fps = frames / max(1e-6, time.time() - start)
        print(f"Haar avg FPS ({benchmark_seconds}s): {avg_fps:.2f}")
        return avg_fps
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_mediapipe(camera_index: int, benchmark_seconds: int) -> float:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam for MediaPipe backend")

    start = time.time()
    frames = 0
    detector_kind, detector = create_mediapipe_detector()
    print(f"MediaPipe detector backend: {detector_kind}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            boxes = detect_mediapipe(frame, detector, detector_kind)
            draw_boxes(frame, boxes, color=(0, 255, 0))

            frames += 1
            elapsed = max(1e-6, time.time() - start)
            fps = frames / elapsed
            cv2.putText(frame, f"MediaPipe FPS: {fps:5.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, "Press Q to quit", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
            cv2.imshow("face_detect - MediaPipe", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
            if benchmark_seconds > 0 and elapsed >= benchmark_seconds:
                break

        avg_fps = frames / max(1e-6, time.time() - start)
        print(f"MediaPipe avg FPS ({benchmark_seconds}s): {avg_fps:.2f}")
        return avg_fps
    finally:
        if hasattr(detector, "close"):
            detector.close()
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Face detection with Haar and MediaPipe backends")
    parser.add_argument("--backend", choices=["haar", "mediapipe", "both"], default="both")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--benchmark-seconds", type=int, default=30)
    args = parser.parse_args()

    if args.backend in ("haar", "both"):
        run_haar(args.camera_index, args.benchmark_seconds)
    if args.backend in ("mediapipe", "both"):
        run_mediapipe(args.camera_index, args.benchmark_seconds)


if __name__ == "__main__":
    main()
