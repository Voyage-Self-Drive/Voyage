import numpy as np
import cv2

cap = cv2.VideoCapture('cars2.mp4')



while cap.isOpened():
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.cv2.Canny(blur, 50, 150)


    cv2.imshow('frame',canny)

    if cv2.waitKey(30) & 0xFF == ord('q')or 0xFF == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()
