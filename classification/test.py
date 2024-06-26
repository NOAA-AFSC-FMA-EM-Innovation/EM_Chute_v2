import configargparse
import data_loader
#import data_loader_class as data_loader
import os
import torch
import models
import utils
from utils import str2bool
import numpy as np
import random
import torch.nn.functional as F
import textwrap
from PIL import Image
from torchvision import datasets, transforms
import yaml
import cv2 as cv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", help = "path to configuration file")
#parser.add_argument("--video_path", help="path to video folder")
#parser.add_argument("--calib_path", help="path to calibration file")
#parser.add_argument("--out_path", help="path to output directory")
args = parser.parse_args()
with open(args.config_path, 'r') as file:
    config = yaml.safe_load(file)

print(config["class_path"])
model_save = torch.load(config["class_path"])

classnames= model_save["classnames"]
prior = model_save["target"]
source_cnt = model_save["source"]
##classnames = ["Arrowtooth Flounder", "Bathyraja Unidentified", "Big Skate", "Darkblotched Rockfish", "Dover Sole",
##              "Dungeness Crab", "English Sole", "Flathead Sole", "Lingcod", "Longnose Skate", "Other", "Pacific Sanddab",
##              "Petrale Sole", "Rex Sole", "Sablefish", "Scallop", "Sea Anemone", "Sea Pen", "Slender Sole", "Spotted Ratfish",
##              "Starfish"]
#[317, 13, 93, 5, 17, 96, 187, 40, 4, 12, 27, 184, 301, 94, 249, 1, 8, 1, 84, 7, 58]
#source_cnt = [2648, 101, 269, 39, 556, 594, 496, 588, 66, 262, 177, 122, 286, 2484, 739, 72, 249, 41, 400, 316, 804]
#[2648, 101, 269, 39, 556, 594, 496, 588, 66, 262, 177, 122, 286, 2484, 739, 72, 249, 41, 400, 316, 804]

model = models.TransferNet(
        21, source_cnt, transfer_loss="lmmd4", base_net="resnet50", max_iter=30, use_bottleneck=True, bottleneck_width=256,).to('cuda')

#prior =[684, 22, 210, 4, 119, 233, 2034, 37, 26, 79, 1, 805, 688, 551, 795, 3, 51, 9, 482, 84, 179]

# [664, 28, 215, 13, 50, 231, 1628, 59, 31, 79, 21, 1002, 952, 496, 799, 4, 35, 10, 494, 85, 200] # All new 87.3
# [684, 22, 210, 4, 119, 233, 2034, 37, 26, 79, 1, 805, 688, 551, 795, 3, 51, 9, 482, 84, 179] # All new 83.0
#[662, 39, 213, 3, 61, 233, 1654, 103, 11, 82, 22, 955, 912, 484, 797, 5, 59, 8, 496, 84, 213] # All new 83.8
#[604, 36, 168, 3, 91, 231, 1573, 136, 80, 124, 42, 800, 1071, 626, 726, 1, 21, 8, 519, 85, 151]# All 2022
#[214, 10, 59, 1, 13, 72, 173, 17, 13, 21, 3, 9, 79, 72, 75, 0, 3, 0, 84, 4, 45] # trip 5
#[72, 1, 70, 1, 6, 37, 84, 19, 19, 8, 0, 303, 207, 55, 226, 0, 3, 0, 75, 1, 29] # trip 6
#[18, 6, 10, 0, 33, 74, 197, 11, 28, 4, 1, 107, 191, 259, 330, 3, 27, 10, 219, 43, 29] #trip 7
# [102, 19, 1, 30, 6, 21, 867, 2, 6, 56, 4, 31, 12, 42, 3, 0, 5, 3, 67, 23, 26] # trip 9
# [181, 26, 28, 0, 29, 35, 538, 143, 24, 14, 1, 286, 58, 139, 93, 0, 3, 1, 206, 5, 21] # trip 10


#model.setprior(prior)
model.load_state_dict(model_save["weights"])#torch.load('output/trip_all_new_83.0.pth'))
#torch.save({"prior":prior,"classnames":classnames,"weights":model.state_dict()},"chute2022.pth")
#print(model.prior)
model.eval()

preprocess = transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])


def get_framenum(img_path):
    framedict = {}
    imgname = {}
    for (dirpath, dirnames, filenames) in os.walk(img_path):
        cnt = 1
        for filename in filenames:
            if(filename[-4:]==".jpg"):
                cnt+=1
                framedict[cnt] = int(filename.strip().split('_')[-3])
                imgname[cnt] = filename
    return framedict, imgname

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

if(not os.path.isabs(config["output_dir"])):
    num_videos = len(os.listdir("../"+config["output_dir"]+"/undistorted"))
    out_dir =  "../"+config["output_dir"]
    vid_folder = "../"+config["output_dir"]+"/undistorted/"
else:
    num_videos = len(os.listdir(config["output_dir"]+"/undistorted"))
    out_dir =  config["output_dir"]
    vid_folder = config["output_dir"]+"/undistorted/"

video_cnt = 1

if (not os.path.exists(out_dir+ "/classification")):
    os.mkdir(out_dir+ "/classification")

outfile_all = open(out_dir + "/classification/total_summary.csv", 'w')

outfile_all.write(
        "vid name, ID, start frame, end frame, avg length, track species, track conf\n")
print("Classification code running now")
for file in os.listdir(out_dir+"/undistorted"):

    color_dict = {}
    
    print("Processing video %s"%(file))
    filename = file[:-4]
    det_path = out_dir+"/detection/"+filename+".txt"
    infile = open(det_path, 'r')
    lines = infile.readlines()
    line_idx = 0
    infile.close()

    video_path = vid_folder + file
    #print(video_path)
    

    out_video_path =out_dir + "/classification/" + file
    
    outfile = open(out_dir + "/classification/"+filename+".csv", 'w')
    outfile2 = open(out_dir + "/classification/"+filename+"_summary.csv", 'w')
    outfile.write(
        "ID, framenum, length, xmin, ymin, xmax, ymax, near border, frame species, frame conf, track species, track conf\n")
    outfile2.write(
        "ID, start frame, end frame, avg length, track species, track conf\n")

    vidcap = cv.VideoCapture(video_path)
    vid_length = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
    success, imgorig = vidcap.read()
    framecnt = 2

    if(config["save_vid"]):
        out = cv.VideoWriter(out_video_path, cv.VideoWriter_fourcc(*'mp4v'), config["fps"], (imgorig.shape[1],imgorig.shape[0]))
        
    results = []
    track_results = []
    track_consensus = []
    avg_length = []
    while (success and line_idx < len(lines)):
        if(framecnt %10 == 0):
            print("classifying video %d/%d: frame %d/%d"%(video_cnt, num_videos,framecnt,vid_length),  end = "\r")
        box = imgorig
        img = cv.cvtColor(imgorig, cv.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        line = lines[line_idx]
        items = line.strip().split(' ')
        frame = int(items[3])
        input_boxes = []
        colors = []
        output_queue = []
        while(frame < framecnt and line_idx < len(lines)):
            line_idx +=1
            if(line_idx < len(lines)):
                line = lines[line_idx]
                items = line.strip().split(' ')
                frame = int(items[3])
        while(frame == framecnt and line_idx < len(lines)):
            trackid = int(items[4])
            xmin = int(items[5])
            ymin = int(items[6])
            width = max(0,int(items[7]))
            height = max(0,int(items[8]))
            
            x1 = xmin
            y1 = ymin
            x2 = x1 + width
            y2 = y1 + height
            img_crop = img.crop((xmin, ymin, xmin + width, ymin + height))
            #img_crop.show()
            img_tensor = preprocess(img_crop)
            img_tensor = img_tensor.unsqueeze(0).to('cuda')
            with torch.no_grad():
                logits = model.predict(img_tensor)

            prob = torch.nn.functional.softmax(logits, dim=1)
            _, preds = torch.max(prob, 1)

            results.append([items[4], framecnt, xmin, xmin + width, ymin, ymin + height, classnames[preds[0]],
            prob[0][preds[0]].cpu().numpy()])
            id = int(items[4])
            while(len(track_results) <= id):
                track_results.append([])
            track_results[id].append([preds[0].cpu().numpy(),prob[0][preds[0]].cpu().numpy(), frame])
            #outfile.write("%s,%d,10.0,%d,%d,%d,%d,0,%s,%f\n" % (
            #items[4], framecnt, xmin, xmin + width, ymin, ymin + height, classnames[preds[0]],
            #prob[0][preds[0]]))
            line_idx += 1
            if(line_idx < len(lines)):
                line = lines[line_idx]
                items = line.strip().split(' ')
                frame = int(items[3])
            if(config["save_vid"]):
                if (trackid not in color_dict.keys()):
                    color_dict[trackid] = (random.randrange(256), random.randrange(256), random.randrange(256))
                    # print(trackid, color_dict[trackid])
                colors.append(color_dict[trackid])
                input_box = [x1, y1, x2, y2]
                input_boxes.append([x1, y1, x2, y2])
                color = color_dict[trackid]
                start_point = (x1, y1)
                end_point = (x2, y2)
                font = cv.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                thickness = 2

                thickness = 2
                box = cv.rectangle(box, start_point, end_point, color, thickness)
                #box = cv2.putText(box, 'gt ID:%03d, gt species %s' % (match_ID, gt_species), start_point, font,
                #                  fontScale, color, thickness, cv2.LINE_AA)
                box = cv.putText(box, 'frame species %s, conf %f' % (classnames[preds[0]], prob[0][preds[0]].cpu().numpy()),

                                  (x1, y1 ), font, fontScale, color, thickness, cv.LINE_AA)
        if(config["save_vid"]):
            out.write(box)
        framecnt +=1
        success, imgorig = vidcap.read()
    out.release()
    for i in range(len(track_results)):
        votes = [0.0]*len(classnames)
        cnt = [0]*len(classnames)
        
        for j in range(len(track_results[i])):
            votes[track_results[i][j][0]]+=track_results[i][j][1]
            cnt[track_results[i][j][0]] += 1

        if(len(track_results[i]) > 0):
            trackid = np.argmax(votes)
            track_consensus.append([trackid, votes[trackid]/cnt[trackid]])
        else:
            track_consensus.append([])
    for i in range(len(results)):
        items = results[i]

        id = int(items[0])

        
        outfile.write("%s,%d,10.0,%d,%d,%d,%d,0,%s,%f, %s, %f\n" % (
        items[0], items[1], items[2],items[4], items[3], items[5], items[6],
        items[7], classnames[track_consensus[id][0]], track_consensus[id][1]))
    outfile.close()


    for i in range(len(track_results)):
        if(len(track_results[i])>0 and len(track_consensus[i])>0):
            outfile_all.write("%s,%s,%d,%d,10.0,%s,%f\n" % ( file,
            i, track_results[i][0][2],track_results[i][-1][2], classnames[track_consensus[i][0]], track_consensus[i][1]))
            outfile2.write("%s,%d,%d,10.0,%s,%f\n" % (
            i, track_results[i][0][2],track_results[i][-1][2], classnames[track_consensus[i][0]], track_consensus[i][1]))
    outfile.close()
    video_cnt +=1

outfile_all.close()
