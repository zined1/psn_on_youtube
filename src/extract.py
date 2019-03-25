from imutils import grab_contours
import numpy as np
import argparse
import imutils
import cv2
import pytesseract
import os


def show(image):
  cv2.imshow("image", image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def get_ROI(image):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  rectKernel = np.ones((3,9),np.uint8)
  tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
  gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
  gradX = np.absolute(gradX)
  minVal, maxVal = (np.min(gradX), np.max(gradX))
  gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
  gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
  thresh = cv2.threshold(gradX, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
  
  cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = grab_contours(cnts)
  
  locs = []

  for (_, c) in enumerate(cnts):
    x, y, w, h = cv2.boundingRect(c)
    locs.append((x,y,w,h))

  return locs

def get_string(img):
  img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
  img = cv2.threshold(cv2.GaussianBlur(img, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
  result = pytesseract.image_to_string(img)
  return result
