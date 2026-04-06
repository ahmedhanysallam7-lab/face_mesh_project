
import mediapipe as mp 

import  cv2
import time

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionConf=.5,trackConf=.5):

        self.mode=mode
        self.maxHands=maxHands
        self.detectionConf=detectionConf

        self.trackConf=trackConf

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
    static_image_mode=self.mode,
    max_num_hands=self.maxHands,
    min_detection_confidence=self.detectionConf,
    min_tracking_confidence=self.trackConf
     )
        
    def findHands(self,frame,draw=True):

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(img_rgb)

            if self.results.multi_hand_landmarks:
               for hand_landmarks in self.results.multi_hand_landmarks:
                  if draw:

                      self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            return frame
    
    def findpos(self,frame,draw=True,handNo=0):
        lmlist=[]
        if self.results.multi_hand_landmarks:

            myhand=self.results.multi_hand_landmarks[handNo]


            for id,lan in enumerate(myhand.landmark):

                      h,w,c=frame.shape
                      cx,cy=int(lan.x*w),int(lan.y*h)

                      lmlist.append([id,cx,cy])

                      #if id==0:
                      if draw:
                         cv2.circle(frame,(cx,cy),21,(255,255,255),7)

        return lmlist





ptime=0
ctime=0
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
detector=handDetector()
while True:
    success, frame = cap.read()
    if not success:
        print("Cannot read frame")
        break

    frame=detector.findHands(frame)

    lmlist=detector.findpos(frame)

    if len(lmlist)!=0:
      print(lmlist[4])



    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime

    cv2.putText(frame,str(int(fps)),(10,50),2,1,(255,0,255),3)

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()