import numpy as np
import cv2
import matplotlib.pyplot as plt


cap = cv2.VideoCapture('cars2.mp4')

def Canny(video):
    gray = cv2.cvtColor(video, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(video, lines):
    frame = np.zeros_like(video)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(frame, (x1, y1), (x2, y2), (255,0,0), 10)
    return frame

def region_of_interest(video):
    height = video.shape[0]
    polygons = np.array([
        [(100, height), (1900,height), (850, 350)]
        ])
    mask = np.zeros_like(video)
    cv2.fillPoly(mask, polygons, (255,255,255))
    masked_video = cv2.bitwise_and(video, mask)
    return masked_video


while cap.isOpened():
    ret, frame = cap.read()
    canny = Canny(frame)
    cropped_video = region_of_interest(canny)
    lines = cv2.HoughLinesP(cropped_video, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap= 5)
    line_video = display_lines(frame, lines)
    combo_video = cv2.addWeighted(frame, 0.8, line_video, 1, 1)
    cv2.imshow('lanes',combo_video)
    if cv2.waitKey(30) & 0xFF == ord('q')or 0xFF == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()