import numpy as np
import cv2
import matplotlib.pyplot as plt


cap = cv2.VideoCapture('cars2.mp4')

def canny(video):
    gray = cv2.cvtColor(video, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny



while cap.isOpened():
    ret, frame = cap.read()
    canny = canny(frame)
    

    cv2.imshow('lanes',canny)
    if cv2.waitKey(30) & 0xFF == ord('q')or 0xFF == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()