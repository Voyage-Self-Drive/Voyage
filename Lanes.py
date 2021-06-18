import numpy as np
import cv2
import matplotlib.pyplot as plt


cap = cv2.VideoCapture('cars2.mp4')

def Canny(video):
    gray = cv2.cvtColor(video, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(video):
    height = video.shape[0]
    polygons = np.array([
        [(300, height), (1800,height), (800, 350)]
        ])
    mask = np.zeros_like(video)
    cv2.fillPoly(mask, polygons, (255,255,255))
    masked_video = cv2.bitwise_and(video, mask)
    return masked_video


while cap.isOpened():
    ret, frame = cap.read()
    canny = Canny(frame)
    cropped_video = region_of_interest(canny)
    
    cv2.imshow('lanes',cropped_video)
    if cv2.waitKey(30) & 0xFF == ord('q')or 0xFF == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()