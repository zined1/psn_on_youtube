import cv2
import numpy as np
import time
import os
import isodate
import pickle
import requests
from pytube import YouTube
from classifier import predict_frame

class Youtube:
  def __init__(self, api_key):
    self.api_key = api_key

  def download(self, video_url, name, output='./'):
    video = YouTube(video_url)
    video.streams.first().download(output_path=output)
    os.rename(video.streams.first().default_filename, name + ".mp4")

  def get_infos(self, code):
    try:
      r = requests.get("https://www.googleapis.com/youtube/v3/videos?part=snippet,contentDetails&id="+ code + "&key="+ self.api_key)
      return r.json()
    except Exception as e:
      print(e)

  def get_duration(self, info_video):
    if "items" in info_video.keys():
      duration_iso = info_video["items"][0]["contentDetails"]["duration"]
      duration_second = isodate.parse_duration(duration_iso)
      return int(duration_second.total_seconds())
  
  def get_name(self, info_video):
    if "items" in info_video.keys():
      return info_video["items"][0]["snippet"]["title"]
    
  def get_last_video(self, channel):
    try:
      r = requests.get("https://www.googleapis.com/youtube/v3/search?order=date&maxResults=1&part=snippet&channelId="+channel+"&key="+self.api_key)
      response = r.json()
      if "items" in response.keys():
        video_id = response["items"][0]["id"]["videoId"]
        return video_id
    except Exception as e:
      print(e) 
    
  def extract_frames(self, code, filter=True, interval_frame=30):
    path = "./" + code + "/"
    info_video = self.get_infos(code)
    duration = self.get_duration(info_video)
    name = self.get_name(info_video)
    fps = 30
    nb_frame = (duration - 1) * fps
    if filter:
      model = pickle.load(open("./model/model.pkl", 'rb'))
    if not os.path.exists(path):
      os.makedirs(path)
      print("Download...")
      self.download("https://www.youtube.com/watch?v="+code, name)
      print("Extracting:", name, "...")
      vidcap = cv2.VideoCapture(name + ".mp4")
      begin = time.time()
      for i in range(0, nb_frame, interval_frame):
        vidcap.set(1, i)
        _, frame = vidcap.read()
        if not filter:
          cv2.imwrite(path + 'frame_'+str(i)+'.jpg', frame)
        elif filter and predict_frame(frame, model):
          cv2.imwrite(path + 'frame_'+str(i)+'.jpg', frame)
      vidcap.release()
      print("Complete", time.time() - begin)
    else:
      print("Video already extracted")
