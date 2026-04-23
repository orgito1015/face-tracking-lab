import cv2, csv, time, numpy as np

class MockTello:
    def __init__(self, log_file="rc_log.csv"):
        self._cap = cv2.VideoCapture(0)
        self._log = open(log_file, 'w', newline='')
        self._writer = csv.writer(self._log)
        self._writer.writerow(['time','lr','fb','ud','yaw','cx','cy','area'])
        self._start = time.time()

    def connect(self): print("[SIM] Connected")
    def get_battery(self): return 100
    def streamon(self): pass
    def streamoff(self): self._cap.release(); self._log.close()
    def takeoff(self): print("[SIM] Takeoff")
    def land(self): print("[SIM] Land")

    def send_rc_control(self, lr, fb, ud, yaw, cx=0, cy=0, area=0):
        t = round(time.time() - self._start, 3)
        self._writer.writerow([t, lr, fb, ud, yaw, cx, cy, area])
        print(f"[t={t:5.2f}s] LR:{lr:+4d} FB:{fb:+4d} UD:{ud:+4d} YAW:{yaw:+4d}")

    class _FR:
        def __init__(self, cap): self._cap = cap
        @property
        def frame(self):
            ret, f = self._cap.read()
            return f if ret else np.zeros((480,640,3),dtype=np.uint8)

    def get_frame_read(self): return MockTello._FR(self._cap)


# PID constants
PID = [0.4, 0.0, 0.1]  # [Kp, Ki, Kd]
MAX_SPEED = 30

FRAME_W, FRAME_H = 640, 480
FACE_TIMEOUT = 3.0

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

def find_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    if len(faces) == 0: return frame, 0, 0, 0
    x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
    cx,cy,area = x+w//2, y+h//2, w*h
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    return frame, cx, cy, area

def track(cx, cy, area, prev_err, dt=0.033):
    err = cx - FRAME_W//2
    yaw = int(np.clip(PID[0]*err + PID[2]*(err-prev_err)/dt,
                      -MAX_SPEED, MAX_SPEED))
    ud_err = cy - FRAME_H//2
    ud = int(np.clip(-PID[0]*ud_err,-MAX_SPEED,MAX_SPEED)) if abs(ud_err)>40 else 0
    fb = -15 if area>14000 else (15 if 0<area<8000 else 0)
    return 0, fb, ud, yaw, err

drone = MockTello()
drone.connect(); drone.streamon(); drone.takeoff()
reader = drone.get_frame_read()
prev_err, last_face = 0, time.time()

try:
    while True:
        frame = cv2.resize(reader.frame, (FRAME_W, FRAME_H))
        frame, cx, cy, area = find_face(frame)
        if area > 0: last_face = time.time()
        elif time.time() - last_face > FACE_TIMEOUT:
            drone.send_rc_control(0,0,0,0)
            cv2.putText(frame,"NO FACE — HOVERING",(20,40),
                       cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.imshow("Tracker",frame); cv2.waitKey(1); continue
        lr,fb,ud,yaw,prev_err = track(cx,cy,area,prev_err)
        drone.send_rc_control(lr,fb,ud,yaw,cx,cy,area)
        cv2.imshow("Tracker",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    drone.land(); drone.streamoff(); cv2.destroyAllWindows()