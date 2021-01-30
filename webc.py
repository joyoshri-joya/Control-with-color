import cv2
import numpy as np
import pyautogui

cap = cv2.VideoCapture(0)

yellow_lower = np.array([22, 93, 0])
yellow_upper = np.array([45, 255, 255])

#lower_white = np.array([0,0,0], dtype=np.uint8)
#upper_white = np.array([0,0,255], dtype=np.uint8)

prev_x = 0

while True:
    ret, frame = cap.read()
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame', gray)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #cv2.imshow('frame', frame)
    mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    #mask = cv2.inRange(hsv, lower_white, upper_white)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame, contours, -1, (0,255,0), 2)
    #cv2.imshow('mask', mask)

    for c in contours:
        area = cv2.contourArea(c)
        if area > 300:
            #cv2.drawContours(frame, contours, -1, (0,255,0), 2)
            #print(area)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
            if x < prev_x:
                pyautogui.press('space')
            #else:
                #print('Doing nothing')
            prev_x = x

    cv2.imshow('frame', frame)
    if cv2.waitKey(10) == ord('j'):
        break

cap.release()

cv2.destroyAllWindows()