import pandas as pd
import numpy as np
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
import pickle

def fd_histogram(img, mask=None):
  bins = 8
  img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  hist  = cv2.calcHist([img], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
  cv2.normalize(hist, hist)
  
  return hist.flatten()

def crop(img):
  height, width = img.shape[:2]
  x = int(0.1 * width)
  y = int(0.2 * height)
  h = height - y
  w = width - x
  return img[y:y+h, x:x+w]

def preprocess_img(img):
  crop_img = crop(img)
  res = cv2.resize(crop_img,(128, 128))
  return res

def predict_frame(img, model):
  frame = preprocess_img(img)
  hist = fd_histogram(frame)
  pred = model.predict([hist])
  if pred[0] == 1:
    return True
  return False

#data = pd.read_csv("./dataset/data.csv")
#classifier = KNeighborsClassifier(n_neighbors=5)  
#classifier.fit(X_train, y_train)

model = pickle.load(open("./model/model.pkl", 'rb'))
