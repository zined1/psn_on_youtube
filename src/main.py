import numpy as np
import argparse
import imutils
import cv2
import pytesseract
import os
import youtube
from imutils import contours
from extract import get_ROI, get_string

ap = argparse.ArgumentParser()
ap.add_argument("-k", "--key", required=True, help="api key")
ap.add_argument("-c", "--channel", required=True, help="channel youtube")
args = vars(ap.parse_args())

if __name__ == "__main__":
  yt = youtube.Youtube(args["key"])
  id_video = yt.get_last_video(args["channel"])
  yt.extract_frames(id_video)
  frames = os.listdir(id_video)
  for frame in frames:
    path = os.path.join(id_video, frame)
    img = cv2.imread(path)
    img = imutils.resize(img, width=500)
    locs = get_ROI(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for (i, (gX, gY, gW, gH)) in enumerate(locs):
      group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 50]
      try:
        code = get_string(group)
        if len(code) >= 13 and len(code) <= 15:
          print(frame, "=>", get_string(group))
      except Exception as e:
        continue
