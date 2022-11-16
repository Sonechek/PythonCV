import cv2
import numpy as np
import os
img = cv2.imread("car.JPG")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = 190

#get threshold image
ret,thresh_img = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

# find contours without approx
contours,_ = cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
max=0
sel_countour=None
for countour in contours:
    if countour.shape[0]>max:
        sel_countour=countour
        max=countour.shape[0]

# calc arclentgh
arclen = cv2.arcLength(sel_countour, True)

# do approx
eps = 0.00001
epsilon = arclen * eps
approx = cv2.approxPolyDP(sel_countour, epsilon, True)

# draw the result
canvas = img.copy()
for pt in approx:
    cv2.circle(canvas, (pt[0][0], pt[0][1]), 7, (0,255,0), -1)

cv2.drawContours(canvas, [approx], -1, (0,0,255), 2, cv2.LINE_AA)

img_contours = np.uint8(np.zeros((img.shape[0],img.shape[1])))
cv2.drawContours(img_contours, [approx], -1, (255,255,255), 1)


cv2.imshow('origin', canvas) # выводим итоговое изображение в окно
cv2.imshow('res', img_contours) # выводим итоговое изображение в окно

cv2.waitKey()
cv2.destroyAllWindows()