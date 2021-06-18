import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.lib.function_base import average


cap = cv2.VideoCapture('cars2.mp4')
car_clf = cv2.CascadeClassifier( 'Haarcascades/haarcascade_car.xml')
## defining the left and right line
def make_coordinates( video, line_parameters):
    slope, intercept = line_parameters
    y1 = video.shape[0]
    y2 = int(y1* (0.5))
    x1 = ((y1 - intercept) / slope)
    x2 = ((y2 - intercept) / slope)
    return np.array([x1,y1,x2,y2])


def find_cars(cam_frame):
    cam_gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
    cars = car_clf.detectMultiScale(cam_gray, 1.1, 2,minSize=(100, 100))
    if len(cars) > 0:
        for (x,y,w,h) in cars:
            ratio = w/float(h)
            #print(w,h)
            cv2.rectangle(cam_frame,(x,y),(x+w,y+h),(255,0,0),10)
            cv2.rectangle(cam_frame, (x, y - 40), (x + w, y), (255,0,0), -2)
            cv2.putText(cam_frame, ' Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            car_gray = cam_gray[y:y+h, x:x+w]
            car_lines = cv2.Canny(car_gray,100,200)
            contours, h = cv2.findContours(car_lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for i, contour in enumerate(contours):
                approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                            #x, y , w, h = cv2.boundingRect(approx)

                if len(approx) ==4:
                    if ratio >= 0.9 and ratio <1.1:
                        cv2.drawContours(cam_frame, contours, i, (255,255,0))

    return cam_frame


## calc the left lane and the right lane
def average_slope_intercept(video, lines):

    ### holder for all lines found with negative slope
    left_fit=[]

    ### holder for all found with positive slope
    right_fit=[]

    ## Loop through all lines to decide negative or positive slope
    for line in lines:

        # start and end codinates of a line
        p = line.reshape(4)
        x1,y1,x2,y2 = line.reshape(4) # (removes uneeded dimenison from (1,4) to (4,)

        # polyfit calculates the slope of the line
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
                    (int(lx1), int(ly1)),
    ]])
    cv2.fillPoly(line_video, poly, (255,0,255))
    cv2.line(line_video, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (255,0,0), 10)
    cv2.line(line_video, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (255,0,0), 10)

    return line_video

## Defining the drivers lane
def region_of_interest(video):
    height = video.shape[0]
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
    cars = find_cars(combo_video)
    cv2.imshow('cars', cars)
    cv2.imshow('lanes', combo_video)
    cv2.imshow('lines', line_video)
    cv2.imshow('region', cropped_video)
    cv2.imshow('canny', canny_image)
    cv2.imshow('original', frame)
    if cv2.waitKey(0) & 0xFF == ord('q')or 0xFF == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()
