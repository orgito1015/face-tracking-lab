"""
Session 1 — Face detection with Haar and MediaPipe backends.

Usage:
    python face_detect.py --backend both --camera-index 0 --benchmark-seconds 30
"""
import argparse
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


Box = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    left_label: str = "",
    right_label: str = "",
) -> np.ndarray:
    """Return a single frame with *left* and *right* placed side-by-side.

    Both panels are resized to the same height.  A thin grey divider is
    inserted between them and optional white panel labels are drawn in the
    top-left corner of each side.
    """
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

    _DIVIDER_GRAY = 80
    divider = np.full((h, 3, 3), _DIVIDER_GRAY, dtype=np.uint8)
    return np.hstack([left, divider, right])


# ---------------------------------------------------------------------------
# MediaPipe helpers
# ---------------------------------------------------------------------------

def _models_dir() -> Path:
    return Path(__file__).resolve().parent / "models"


def ensure_mediapipe_face_model() -> str:
    model_path = _models_dir() / "blaze_face_short_range.tflite"
    if model_path.exists():
        return str(model_path)

    _models_dir().mkdir(parents=True, exist_ok=True)
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
            "Could not download MediaPipe model. Check your internet connection or "
            "manually place blaze_face_short_range.tflite in face_lab/models/."
        ) from exc


def _get_mediapipe_kind() -> str:
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
        return "legacy"
    if hasattr(mp, "tasks"):
        return "tasks"
    raise RuntimeError(
        "MediaPipe face detection API is unavailable. "
        "Reinstall with: pip install --upgrade mediapipe"
    )


def create_mediapipe_detector():
    kind = _get_mediapipe_kind()
    if kind == "legacy":
        detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        return kind, detector

    # Tasks API path — imports are gated here to avoid ImportError on legacy builds
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


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def detect_haar(frame, cascade: cv2.CascadeClassifier) -> List[Box]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def detect_mediapipe(frame, detector, kind: str) -> List[Box]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if kind == "legacy":
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

    # Tasks API — bounding_box is already in pixel coordinates
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    if not result.detections:
        return []
    return [
        (int(det.bounding_box.origin_x), int(det.bounding_box.origin_y),
         int(det.bounding_box.width), int(det.bounding_box.height))
        for det in result.detections
    ]


def draw_boxes(frame, boxes: List[Box], color=(0, 255, 0)) -> None:
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


# ---------------------------------------------------------------------------
# Backend runners
# ---------------------------------------------------------------------------

def _open_camera(camera_index: int) -> cv2.VideoCapture:
    if sys.platform == "win32":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
    return cap


def run_haar(camera_index: int, benchmark_seconds: int) -> float:
    cap = _open_camera(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam for Haar backend.")

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        raise RuntimeError("Could not load haarcascade_frontalface_default.xml.")

    start = time.time()
    frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            raw = frame.copy()
            boxes = detect_haar(frame, cascade)
            draw_boxes(frame, boxes, color=(0, 255, 255))

            frames += 1
            elapsed = max(time.time() - start, 1e-6)
            fps = frames / elapsed

            cv2.putText(frame, f"Haar  FPS: {fps:.1f}  Faces: {len(boxes)}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

            combined = _side_by_side(raw, frame, "Camera", "Haar Detection")
            cv2.imshow("face_detect - Haar", combined)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
            if benchmark_seconds > 0 and elapsed >= benchmark_seconds:
                break

        avg_fps = frames / max(time.time() - start, 1e-6)
        print(f"Haar avg FPS ({benchmark_seconds}s): {avg_fps:.2f}")
        return avg_fps
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_mediapipe(camera_index: int, benchmark_seconds: int) -> float:
    cap = _open_camera(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam for MediaPipe backend.")

    kind, detector = create_mediapipe_detector()
    print(f"MediaPipe detector backend: {kind}")

    start = time.time()
    frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            raw = frame.copy()
            boxes = detect_mediapipe(frame, detector, kind)
            draw_boxes(frame, boxes, color=(0, 255, 0))

            frames += 1
            elapsed = max(time.time() - start, 1e-6)
            fps = frames / elapsed

            cv2.putText(frame, f"MediaPipe  FPS: {fps:.1f}  Faces: {len(boxes)}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            combined = _side_by_side(raw, frame, "Camera", "MediaPipe Detection")
            cv2.imshow("face_detect - MediaPipe", combined)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
            if benchmark_seconds > 0 and elapsed >= benchmark_seconds:
                break

        avg_fps = frames / max(time.time() - start, 1e-6)
        print(f"MediaPipe avg FPS ({benchmark_seconds}s): {avg_fps:.2f}")
        return avg_fps
    finally:
        if hasattr(detector, "close"):
            detector.close()
        cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_both_combined(camera_index: int, benchmark_seconds: int) -> None:
    """Run Haar and MediaPipe detectors simultaneously on every frame.

    Displays a single side-by-side window: Haar result on the left,
    MediaPipe result on the right.
    """
    cap = _open_camera(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam for combined backend.")

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        raise RuntimeError("Could not load haarcascade_frontalface_default.xml.")

    kind, mp_detector = create_mediapipe_detector()
    print(f"MediaPipe detector backend: {kind}")

    start = time.time()
    frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            haar_frame = frame.copy()
            mp_frame = frame.copy()

            haar_boxes = detect_haar(haar_frame, cascade)
            draw_boxes(haar_frame, haar_boxes, color=(0, 255, 255))

            mp_boxes = detect_mediapipe(mp_frame, mp_detector, kind)
            draw_boxes(mp_frame, mp_boxes, color=(0, 255, 0))

            frames += 1
            elapsed = max(time.time() - start, 1e-6)
            fps = frames / elapsed

            cv2.putText(
                haar_frame,
                f"Haar  FPS: {fps:.1f}  Faces: {len(haar_boxes)}",
                (8, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2,
            )
            cv2.putText(
                mp_frame,
                f"MediaPipe  FPS: {fps:.1f}  Faces: {len(mp_boxes)}",
                (8, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,
            )

            combined = _side_by_side(haar_frame, mp_frame, "Haar", "MediaPipe")
            cv2.imshow("face_detect - Haar vs MediaPipe", combined)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
            if benchmark_seconds > 0 and elapsed >= benchmark_seconds:
                break

        avg_fps = frames / max(time.time() - start, 1e-6)
        print(f"Combined avg FPS ({benchmark_seconds}s): {avg_fps:.2f}")
    finally:
        if hasattr(mp_detector, "close"):
            mp_detector.close()
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Face detection: Haar and/or MediaPipe backends"
    )
    parser.add_argument(
        "--backend", choices=["haar", "mediapipe", "both"], default="both",
        help="Detection backend to use (default: both)"
    )
    parser.add_argument("--camera-index", type=int, default=0,
                        help="Webcam index (default: 0)")
    parser.add_argument("--benchmark-seconds", type=int, default=30,
                        help="Stop after N seconds and print avg FPS (default: 30)")
    args = parser.parse_args()

    if args.backend == "both":
        run_both_combined(args.camera_index, args.benchmark_seconds)
    elif args.backend == "haar":
        run_haar(args.camera_index, args.benchmark_seconds)
    elif args.backend == "mediapipe":
        run_mediapipe(args.camera_index, args.benchmark_seconds)


if __name__ == "__main__":
    main()
