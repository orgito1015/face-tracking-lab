import cv2, time

cap = cv2.VideoCapture(0)
frame_count, start = 0, time.time()

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1
    fps = frame_count / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(frame, time.strftime("%H:%M:%S"), (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,200,0), 2)
    cv2.imshow("Webcam Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
