import numpy as np
import cv2

#Description missing

cap = cv2.VideoCapture('cars2.mp4')


def make_coordinates(video, line_parameters):
    slope, intercept = line_parameters
    y1 = video.shape[0]
    y2 = int(y1 * 0.5)
    x1 = ((y1 - intercept) / slope)
    x2 = ((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(video, res_lines):
    left_fit, right_fit = [], []
    for line in res_lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(video, left_fit_average)
    right_line = make_coordinates(video, right_fit_average)
    return np.array([left_line, right_line])


def canny(video):
    gray = cv2.cvtColor(video, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    result = cv2.Canny(blur, 50, 150)
    return result


def display_lines(video, res_lines):
    res = np.zeros_like(video)
    if res_lines is not None:
        for x1, y1, x2, y2 in res_lines:
            cv2.line(res, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 10)
    return res


def region_of_interest(video):
    height = video.shape[0]
    polygons = np.array([[(820, 340), (960, 340), (1280, 520), (1280, 720), (125, 720)]])
    mask = np.zeros_like(video)
    cv2.fillPoly(mask, polygons, (255, 255, 255))
    masked_video = cv2.bitwise_and(video, mask)
    return masked_video


def fill_area(video, res_lines):
    res = np.zeros_like(video)
    if res_lines is not None:
        top_left, top_right = (res_lines[0][0], res_lines[0][1]), (res_lines[0][2], res_lines[0][3])
        bottom_left, bottom_right = (res_lines[1][0], res_lines[1][1]), (res_lines[1][2], res_lines[1][3])
        points = np.array([top_left, top_right, bottom_right, bottom_left]).astype(int)
        res = cv2.fillPoly(img=video, pts=[points], color=(0, 250, 0))
        for x1, y1, x2, y2 in res_lines:
            res = cv2.circle(res, (int(x1), int(y1)), radius=10, color=(0, 255, 0), thickness=-1)
            res = cv2.circle(res, (int(x2), int(y2)), radius=10, color=(0, 255, 0), thickness=-1)

    return res


while cap.isOpened():
    ret, frame = cap.read()
    canny_image = canny(frame)
    cropped_video = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_video, rho=2, theta=np.pi/180, threshold=100,
                            lines=np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_video = display_lines(frame, averaged_lines)
    area_video = fill_area(line_video, averaged_lines)

    combo_video = cv2.addWeighted(frame, 0.8, area_video, 1, 1)

    cv2.imshow('Lanes', combo_video)
    if cv2.waitKey(30) & 0xFF == ord('q') or 0xFF == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()
