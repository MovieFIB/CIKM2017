import os
import json


pathPrefix = '/path/to/video/file'
videofiles = json.load(open('videofile.json', 'r'))
videofilepath = open('videofilepath.txt', 'w')
for videofile in videofiles:
    video_path = os.path.join(pathPrefix, videofile)
    videofilepath.write(video_path+'\n')
videofilepath.close()

