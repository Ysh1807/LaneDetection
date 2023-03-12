import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    c = cv2.Canny(blur, 50, 150)
    return c


def reg_interest(image):
    triangle = np.array([
        [(200, 720), (800, 720), (625, 522)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img


def draw_lines(image, lines):

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return image


img = cv2.imread('C:\Resources\img.jpg')
img = cv2.resize(img, (1280, 720))
lane_img = np.copy(img)
canny = canny(lane_img)
reg = reg_interest(canny)
detected_lines = cv2.HoughLinesP(reg, 2, np.pi/180, 100, np.array([]), 40, 5)
final_image = draw_lines(lane_img, detected_lines)

cv2.imshow('Result', final_image)
cv2.imshow('Image', reg)
cv2.waitKey(0)
