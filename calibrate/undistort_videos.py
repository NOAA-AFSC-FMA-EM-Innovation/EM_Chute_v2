import cv2 as cv
import numpy as np
import xml.etree.ElementTree as ET
import textwrap
import argparse
import os
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", help = "path to configuration file")
#parser.add_argument("--video_path", help="path to video folder")
#parser.add_argument("--calib_path", help="path to calibration file")
#parser.add_argument("--out_path", help="path to output directory")
args = parser.parse_args()

with open(args.config_path, 'r') as file:
    config = yaml.safe_load(file)

def parse_array(rows, cols, data):
    data = data.replace('\n', '').strip()
    data = textwrap.dedent(data)
    array = np.zeros((rows, cols))
    row = 0
    col = 0
    for item in data.split(" "):
        if(item !=''):
            array[row,col] = float(item)
            col+=1
            if(col%cols == 0):
                col = 0
                row +=1
    return array
tree = ET.parse(config["calib_path"])
root = tree.getroot()
mat_element = root.find("CameraParamBEV/CameraParam/intrinsic_matrix")
rows = int(mat_element.find("rows").text)
cols = int(mat_element.find("cols").text)
calib = parse_array(rows, cols, mat_element.find("data").text)

mat_element = root.find("CameraParamBEV/CameraParam/distort_coeff")
rows = int(mat_element.find("rows").text)
cols = int(mat_element.find("cols").text)
dist = parse_array(rows, cols, mat_element.find("data").text)

mat_element = root.find("CameraParamBEV/R_BEV")
rows = int(mat_element.find("rows").text)
cols = int(mat_element.find("cols").text)
rbev = parse_array(rows, cols, mat_element.find("data").text)

mat_element = root.find("CameraParamBEV/K_BEV")
rows = int(mat_element.find("rows").text)
cols = int(mat_element.find("cols").text)
kbev = parse_array(rows, cols, mat_element.find("data").text)

element = root.find("CameraParamBEV/alphaBEV")
alpha = float(element.text)

element = root.find("CameraParamBEV/szBEV")
items = element.text.strip().split(" ")
bevsize = (int(items[0]), int(items[1]))

map1, map2 = cv.initUndistortRectifyMap(calib, dist, rbev, kbev, bevsize, cv.CV_32FC1)

if(not os.path.exists(config["output_dir"])):
    os.mkdir(config["output_dir"])
    os.mkdir(config["output_dir"] + "/undistorted")

print("Undistortion code running now")
video_cnt = 1
num_videos = len(os.listdir(config["input_dir"]))
for vidname in os.listdir(config["input_dir"]):
    print("Processing video %s"%(vidname))
    
    video_path = config["input_dir"] + "/" + vidname
    out_path = config["output_dir"] + "/undistorted/" + vidname
    out = cv.VideoWriter(out_path,cv.VideoWriter_fourcc(*'mp4v'), config["fps"], bevsize)

    vidcap = cv.VideoCapture(video_path)
    vid_length = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
    success, img = vidcap.read()
    framecnt = 1
    while(success):
        if(framecnt %10 == 0):
            print("Undistorting video %d/%d: frame %d/%d"%(video_cnt, num_videos,framecnt,vid_length),  end = "\r")
        undistimg = cv.remap(img, map1, map2, cv.INTER_LINEAR)
        success, img = vidcap.read()
        out.write(undistimg)
        framecnt+=1

    vidcap.release()
    video_cnt +=1
    out.release()
