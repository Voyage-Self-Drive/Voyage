import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.lib.function_base import average


cap = cv2.VideoCapture('cars2.mp4')

## defining the left and right line
def make_coordinates( video, line_parameters):
    print(f"LINE PARAMAS {line_parameters}")
    print(line_parameters.shape)
    slope, intercept = line_parameters
    y1 = video.shape[0]
    y2 = int(y1* (0.5))
    x1 = ((y1 - intercept) / slope)
    x2 = ((y2 - intercept) / slope)
    return np.array([x1,y1,x2,y2])


## calc the left lane and the right lane
def average_slope_intercept(video, lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        print(x1,y1,x2,y2)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis = 0)
    print(type(left_fit_average))
    right_fit_average = np.average( right_fit, axis= 0)
    left_line = make_coordinates(video, left_fit_average)
    right_line = make_coordinates(video, right_fit_average)
    return np.array([left_line, right_line])

## preprossecing the video, grayscaling, blur and canny
def Canny(video):
    gray = cv2.cvtColor(video, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

## Definining the left and right lane/line and draw them
def display_lines(video, lines):
    line_video = np.zeros_like(video)
    if lines is not None:
        for x1, y1, x2, y2  in lines:
            #print(x1, y1 ,x2, y2)
            cv2.line(line_video, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 10)
    return line_video

## Defining the drivers lane
def region_of_interest(video):
    height = video.shape[0]
    print(height)
    polygons = np.array([
        [(820, 340), (960,340), (1280, 520), (1280,720), (125,720)]])
    mask = np.zeros_like(video)
    cv2.fillPoly(mask, polygons, (255,255,255))
    masked_video = cv2.bitwise_and(video, mask)
    return masked_video


## Showing the Video
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
