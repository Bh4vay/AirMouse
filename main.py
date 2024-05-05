import cv2
import numpy as np
import mediapipe as mp
import time
import pyautogui

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
pTime = 0
handDetector = mp.solutions.hands.Hands(max_num_hands=1)
drawing_utils = mp.solutions.drawing_utils
screen_width,screen_height = pyautogui.size()
index_y = 0

while True:
    ret, img = cap.read()
    img = cv2.flip(img,1)
    img_height, img_width , _ = img.shape
    rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    # Detecting the hand landmarks
    output = handDetector.process(rgb_img)
    hand_lndmrks = output.multi_hand_landmarks
    if hand_lndmrks:
        for hand_lndmrk in hand_lndmrks:
            drawing_utils.draw_landmarks(img,hand_lndmrk)
            landmarks = hand_lndmrk.landmark
            for id,landmark in enumerate(landmarks):
                x = int(landmark.x * img_width)
                y = int(landmark.y * img_height)
                print(int(x),int(y))
                
                # getting index finger's id
                if id == 8:
                    cv2.circle(img=img,center=(x,y),radius=10,color=(0,255,0),thickness=2)
                    
                    # to adjust according to whole screen, not just the frame
                    index_x = screen_width/img_width*x
                    index_y = screen_height/img_height*y
                    pyautogui.moveTo(index_x,index_y)
                    
                # getting thumb's id
                if id == 4:
                    cv2.circle(img=img,center=(x,y),radius=10,color=(0,255,0),thickness=2)
                    thumb_x = screen_width/img_width*x
                    thumb_y = screen_height/img_height*y
                    
                    # if distance b/w them less, then click
                    if abs(index_y - thumb_y) < 20:
                        pyautogui.click()
                        pyautogui.sleep(1)
    
    # Frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
    
    
    # Display
    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
    
cap.release()
    