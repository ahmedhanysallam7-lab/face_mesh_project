import cv2
import mediapipe as mp
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import time
import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

app = FastAPI()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0  # 👈 أسرع
)

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)
def generate_frames():
    while True:
        success, frame = cap.read()

        if not success:
            print("❌ camera error")
            continue   # 👈 أهم تعديل

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(0.03)  # 👈 يمنع التهنيج
        

@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )
"""
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
ptime=0
ctime=0
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        print("Cannot read frame")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            for id,lan in enumerate(hand_landmarks.landmark):
               # print(id,lan)
                h,w,c=frame.shape
                cx,cy=int(lan.x*w),int(lan.y*h)

                print(id,cx,cy)

                if id==0:
                    cv2.circle(frame,(cx,cy),21,(255,255,255),7)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime

    cv2.putText(frame,str(int(fps)),(10,50),2,1,(255,0,255),3)

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()"""