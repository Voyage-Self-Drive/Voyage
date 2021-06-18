import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.lib.function_base import average


cap = cv2.VideoCapture('cars2.mp4')


def make_coordinates( video, line_parameters):
    slope, intercept = line_parameters
    y1 = video.shape[0]
    y2 = int(y1* (3/5))
    x1 = ((y1 - intercept) / slope)
    x2 = ((y2 - intercept) / slope)
    return np.array([x1,y1,x2,y2])



def average_slope_intercept(video, lines):
    left_fit=[]
    right_fit=[]
    for line in lines: 
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
        
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average( right_fit, axis= 0)
    left_line = make_coordinates(video, left_fit_average)
    right_line = make_coordinates(video, right_fit_average)
    return np.array([left_line, right_line])


def Canny(video):
    gray = cv2.cvtColor(video, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(video, lines):
    line_video = np.zeros_like(video)
    if lines is not None:
        for x1, y1, x2, y2  in lines:
            print(x1, y1 ,x2, y2)
            cv2.line(line_video, (x1, y1), (x2, y2), (255,0,0), 10)
    return line_video

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
    canny_image = Canny(frame)
    cropped_video = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_video, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap= 5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_video = display_lines(frame, averaged_lines)
    combo_video = cv2.addWeighted(frame, 0.8, line_video, 1, 1)


    cv2.imshow('lanes', combo_video)
    if cv2.waitKey(30) & 0xFF == ord('q')or 0xFF == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()