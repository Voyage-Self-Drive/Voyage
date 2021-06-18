import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.lib.function_base import average


cap = cv2.VideoCapture('../c2.mp4')

## defining the left and right line
def make_coordinates( video, line_parameters):
    slope, intercept = line_parameters
    y1 = video.shape[0]
    y2 = int(y1* (0.5))
    x1 = ((y1 - intercept) / slope)
    x2 = ((y2 - intercept) / slope)
    return np.array([x1,y1,x2,y2])


## calc the left lane and the right lane
def average_slope_intercept(video, lines):

    ### holder for all lines found with negative slope
    left_fit=[]

    ### holder for all found with positive slope
    right_fit=[]

    ## Loop through all lines to decide negative or positive slope
    print(f"LENGTH OF LINES {len(lines)}")
    for line in lines:


        # start and end codinates of a line
        p = line.reshape(4)
        x1,y1,x2,y2 = line.reshape(4) # (removes uneeded dimenison from (1,4) to (4,)
        print(x1,y1,x2,y2)
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)


        # polyfit calculates the slope of the line
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    print(len(left_fit))
    print(len(right_fit))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average( right_fit, axis= 0)
    print(f"{left_fit_average}/ {right_fit_average}")

    if len(left_fit) == 0 or len(right_fit) == 0:
        return "fuck"
    else:
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
    left_line = lines[0]
    right_line = lines[1]
    lx1, ly1, lx2, ly2 = left_line
    rx1, ry1, rx2, ry2 = right_line
    poly = np.array([[
                    (int(lx1), int(ly1)),
                    (int(rx1), int(ry1)),
                    (int(rx2), int(ry2)),
                    (int(lx2), int(ly2)),
    ]])
    print(poly)
    cv2.fillPoly(line_video, poly, (255,0,255))
    cv2.line(line_video, (min(int(rx1),1280), min(int(ry1),720)), (min(int(rx2),1280), min(int(ry2), 720)), (255,0,0), 10)
    cv2.line(line_video, (min(int(lx1),1280), min(int(ly1),720)), (min(int(lx2),1280), min(int(ly2),720)), (255,0,0), 10)
    return line_video

## Defining the drivers lane
def region_of_interest(video):
    height = video.shape[0]
    polygons = np.array([
        [(595,450),
        (757,431),
        (1126, 636),
        (117, 655),]])
    mask = np.zeros_like(video)
    cv2.fillPoly(mask, polygons, (255,255,255))
    masked_video = cv2.bitwise_and(video, mask)
    return masked_video


## Showing the Video
while cap.isOpened():
    ret, frame = cap.read()
    print(f"frame shape = {frame.shape}")
    canny_image = Canny(frame)

    cropped_video = region_of_interest(canny_image)
    cv2.imshow('region', cropped_video)
    lines = cv2.HoughLinesP(cropped_video, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap= 5)
    averaged_lines = average_slope_intercept(frame, lines)
    if averaged_lines == "fuck":
        continue
    else:
        line_video = display_lines(frame, averaged_lines)
        combo_video = cv2.addWeighted(frame, 0.8, line_video, 1, 1)

    cv2.imshow('lanes', combo_video)
    cv2.imshow('lines', line_video)

    cv2.imshow('canny', canny_image)
    cv2.imshow('original', frame)
    if cv2.waitKey(30) & 0xFF == ord('q')or 0xFF == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()
