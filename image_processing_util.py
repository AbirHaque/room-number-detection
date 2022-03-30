import cv2 as cv
import imutils

def get_edges(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(5,5),0)
    edges = cv.Canny(blur,150, 255,255)
    return edges

def get_quadrilaterals(edges):
    quadrilaterals = []
    #Localize object
    contours = cv.findContours(edges.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    for cnt in contours:
        epsilon = cv.arcLength(cnt,True)
        approx = cv.approxPolyDP(cnt,0.1*epsilon,True)
        if len(approx)==4:
            quadrilaterals.append(approx)
    return quadrilaterals