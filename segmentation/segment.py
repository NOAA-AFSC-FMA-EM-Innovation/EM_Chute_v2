import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import time
import random
from math import atan2, cos, sin, sqrt, pi
import argparse
import yaml
import matplotlib.animation as animation
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", help = "path to configuration file")
args = parser.parse_args()


with open(args.config_path, 'r') as file:
    config = yaml.safe_load(file)

tree = ET.parse("../"+config["calib_path"])
root = tree.getroot()
element = root.find("CameraParamBEV/mBEV")
mBEV = float(element.text)

mean = 0
eigenvectors = []
eigenvalues = []
perform_pca = True
direc = 1
midline_pts = config["midline_pts"]
kernel_size = 8
find_kernel_size = True


def show_mask(mask, ax, random_color=False, color = (0,0,0)):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.concatenate([np.array(color)/256, np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

def getHeadpoint(pts, cent, eigenvectors, direc):
    maxmag = 0
    maxidx = -1
    for i in range(pts.shape[0]):
       mag = (pts[i,0]-cent[0])*eigenvectors[0,0]*direc + (pts[i][1]-cent[1])*eigenvectors[0,1]*direc
       if(mag > maxmag):
            maxmag = mag
            maxidx = i
    return maxidx

def getTailpoint(pts, cent, eigenvectors, direc):
    maxmag = 0
    maxidx1 = -1
    direct = direc*-1
    for i in range(pts.shape[0]):
       mag = (pts[i,0]-cent[0])*eigenvectors[0,0]*direct + (pts[i][1]-cent[1])*eigenvectors[0,1]*direct
       #mag += abs(((pts[i,0]-cent[0])*eigenvectors[1,0] + (pts[i][1]-cent[1])*eigenvectors[1,1]))*0.5

       if(mag > maxmag):
            maxmag = mag
            maxidx1 = i

    maxmag = 0
    maxidx2 = -1
    for i in range(pts.shape[0]):
        mag = (pts[i,0]-cent[0])*eigenvectors[0,0]*direct + (pts[i][1]-cent[1])*eigenvectors[0,1]*direct
        mag += abs(((pts[i,0]-cent[0])*eigenvectors[1,0] + (pts[i][1]-cent[1])*eigenvectors[1,1])
                   - ((pts[maxidx1,0]-cent[0])*eigenvectors[1,0] + (pts[maxidx1][1]-cent[1])*eigenvectors[1,1]))*0.5
        if(mag > maxmag):
            maxmag = mag
            maxidx2 = i
    #print(maxidx1, maxidx2)
    return maxidx1, maxidx2


def getPts(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    tail_pts = []
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    global perform_pca, mean, eigenvectors, eigenvalues, direc
    if(perform_pca):
        # Perform PCA analysis
        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
        perform_pca = False
    

    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))

    if(perform_pca):
        mass1 = 0
        mass2 = 0
        for i in range(data_pts.shape[0]):
            proj1 = (data_pts[i,0]-cntr[0])*eigenvectors[0,0]+(data_pts[i,1]-cntr[1])*eigenvectors[0,1]
            proj2 = (data_pts[i,0]-cntr[0])*eigenvectors[1,0]+(data_pts[i,1]-cntr[1])*eigenvectors[1,1]
            if(proj1 > 0):
                mass1 += abs(proj2)
            else:
                mass2 += abs(proj2)
        direc = 1
        if(mass2 > mass1):
            direc = -1
    #print(direc)
    hpt = getHeadpoint(data_pts, cntr, eigenvectors, direc)
    tpt1, tpt2 = getTailpoint(data_pts, cntr, eigenvectors, direc)
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    cv2.circle(img, (int(data_pts[hpt,0]), int(data_pts[hpt,1])) , 3, (255, 0, 255), 2)
    cv2.circle(img, (int(data_pts[tpt1,0]), int(data_pts[tpt1,1])), 3, (255, 0, 255), 2)
    #cv2.circle(img, (int(data_pts[tpt2,0]), int(data_pts[tpt2,1])), 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0]*direc, cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0]*direc)
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians

    return [int(data_pts[hpt,0]), int(data_pts[hpt,1])], [int(data_pts[tpt1,0]), int(data_pts[tpt1,1])]

def getMidline(mask_show):
    blank = np.zeros(mask_show.shape, dtype = np.uint8)
    contours, _ = cv2.findContours(mask_show, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    maxidx = 0
    maxarea = 0
    minsize = 0
    global find_kernel_size, kernel_size

    for i in range(len(contours)):
        contour = contours[i]
        area = cv2.contourArea(contour)
        if(area > maxarea):
            maxarea = area
            maxidx = i
            x,y,w,h = cv2.boundingRect(contour)
    if(maxarea > 0):
        if(find_kernel_size):
            kernel_size = max(1, int(min(w,h)/midline_pts))
            find_kernel_size = False

        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        erosion = cv2.erode(mask_show,kernel,iterations = 1)
        hpt, tpt = getPts(contours[maxidx], mask_show)
        hpts, tpts = getMidline(erosion)
        hpts.insert(0,hpt)
        tpts.append(tpt)
    else:
        hpts = []
        tpts = []
    #print(hpts)
    return hpts, tpts

def getLength(pts):
    length = 0.0
    for i in range(len(pts)-1):
        length += ((pts[i+1][0]-pts[i][0])**2 + (pts[i+1][1]-pts[i][1])**2)**0.5

    return length
        
    
#image = cv2.imread('images/truck.jpg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
model_paths = {"vit_l": "sam_vit_l_0b3195.pth",
               "vit_b": "sam_vit_b_01ec64.pth",
               "vit_h": "sam_vit_h_4b8939.pth"}

#sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_"+config["segment_model"]
sam_checkpoint = model_paths[model_type]
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
##input_box = np.array([425, 600, 700, 875])
##masks, _, _ = predictor.predict(
##    point_coords=None,
##    point_labels=None,
##    box=input_box[None, :],
##    multimask_output=False,
##)

fig = plt.figure(figsize=(10,10))
#fig.margins(0,0)

if(not os.path.isabs(config["output_dir"])):
    video_dir = "../"+config["output_dir"]+"/undistorted"
    det_dir =  "../"+config["output_dir"]+"/detection"
    class_dir =  "../"+config["output_dir"]+"/classification"
    #share_dir = "F:/noaa/chute_2022/results_trip_%d_share/"%(trip)
    out_dir =  "../"+config["output_dir"]+"/segmentation"
else:
    video_dir = config["output_dir"]+"/undistorted"
    det_dir =  config["output_dir"]+"/detection"
    class_dir =  config["output_dir"]+"/classification"
    #share_dir = "F:/noaa/chute_2022/results_trip_%d_share/"%(trip)
    out_dir =  config["output_dir"]+"/segmentation"
if(not os.path.exists(out_dir)):
    os.mkdir(out_dir)
cnt = 0
total_time=0

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
thickness = 2
num_videos = len(os.listdir(video_dir))
vid_cnt = 1
for path in os.listdir(video_dir):
    name = path[:-4]
    detections = {}
    color_dict = {}
    print("Segmenting video %s"%(path))
    video_path =video_dir + "/" + path
    vidcap = cv2.VideoCapture(video_path)

    success, img = vidcap.read()
    framecnt = 2
    im_size = (img.shape[1],img.shape[0])
    out = cv2.VideoWriter(out_dir + "/" + path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (img.shape[1],img.shape[0]))
    f2 = class_dir + "/" + name + ".csv"
    f3 = out_dir + "/" + name + ".csv"
    f4 = out_dir + "/"+name
    #if(not os.path.exists(f4)):
        #os.mkdir(f4)
    file = open(f2, 'r')
    lines = file.readlines()
    file.close()
    outfile = open(f3,'w')
    outfile.write(lines[0])
    total_time = 0
    while(success ):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        box = img.copy()
        mask = img.copy()
        input_boxes = []
        colors = []
        output_queue = []
        st = time.time()
        for line in lines[1:]:
            items = line.strip().split(',')
            if (int(items[1]) == framecnt):
                output_queue.append(line)
                x1 = int(items[3])
                y1 = int(items[4])
                x2 = int(items[5])
                y2 = int(items[6])
                gt_species = items[-1]
                track_species = items[10]
                track_conf = float(items[11])
                frame_species = items[8]
                frame_conf = float(items[9])
                trackid = int(items[0])
                #match_ID = int(items[-2])
                if (trackid not in color_dict.keys()):
                    color_dict[trackid] = (random.randrange(256), random.randrange(256), random.randrange(256))
                    # print(trackid, color_dict[trackid])
                colors.append(color_dict[trackid])
                input_box = [x1, y1, x2, y2]
                input_boxes.append([x1, y1, x2, y2])
                show_box(input_box, fig.gca())
                color = color_dict[trackid]
                start_point = (x1, y1)
                end_point = (x2, y2)
                thickness = 2
                box = cv2.rectangle(box, start_point, end_point, color, thickness)
                #box = cv2.putText(box, 'gt ID:%03d, gt species %s' % (match_ID, gt_species), start_point, font,
                #                  fontScale, color, thickness, cv2.LINE_AA)
                box = cv2.putText(box,
                                  'track ID:%03d, track species %s, conf %f' % (trackid, track_species, track_conf),
                                  (x1, y1 - 30), font, fontScale, color, thickness, cv2.LINE_AA)
                box = cv2.putText(box, '              frame species %s, conf %f' % (frame_species, frame_conf),
                                  (x1, y1 ), font, fontScale, color, thickness, cv2.LINE_AA)
        fig.gca().imshow(box)
        # print(colors)
        if (len(input_boxes) > 0 and config["segment"]):
            predictor.set_image(img)
            input_boxes = torch.tensor(input_boxes, device=predictor.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, img.shape[:2])
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            index = 0
            for mask in masks:
                mask_show = mask.squeeze().cpu().numpy().astype(np.uint8) * 255

                imgshow = cv2.cvtColor(mask_show, cv2.COLOR_GRAY2RGB)

                perform_pca = True
                find_kernel_size = True
                hpts, tpts = getMidline(mask_show)

                hpts.extend(tpts)
                length = getLength(hpts)*0.1/mBEV
                items = output_queue[index].strip().split(',')
                outfile.write("%s,%s,%0.4f,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%(items[0],items[1],length,items[3],
                                                                        items[4],items[5],items[6],items[7],items[8],items[9],items[10],items[11]))
                for i in range(len(hpts) - 1):
                    box = cv2.line(box, (hpts[i][0], hpts[i][1]), (hpts[i + 1][0], hpts[i + 1][1]), (0, 0, 0),
                                   thickness * 2)
                index +=1
            index = 0
            fig.gca().imshow(box)
            for mask in masks:
                show_mask(mask.cpu().numpy(), fig.gca(), random_color=False, color=colors[index])
                index += 1
        success, img = vidcap.read()
        et = time.time()
        elapsed_time = et-st
        total_time +=elapsed_time
        average = total_time/framecnt
        print(" Segmenting video %d/%d: frame %d, avg seconds per frame: %0.3f"%(vid_cnt,num_videos, framecnt, average),end = '\r')
        framecnt +=1
        # cv2.imwrite(f3, box)
        fig.gca().axis('off')
        fig.canvas.draw()
        #outimg  = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        #outimg = outimg.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #outimg = cv2.cvtColor(outimg, cv2.COLOR_RGB2BGR)
        #cv2.imwrite(f4+"/frame%d.jpg"%(framecnt-1), outimg)
        #out.write(outimg)
        fig.gca().axis('off')
        fig.savefig(out_dir+"/temp.jpg", bbox_inches='tight')

        savimg = cv2.imread(out_dir+"/temp.jpg")
        savimg = cv2.resize(savimg,(im_size))
        out.write(savimg)

        fig.clf()
    outfile.close()
    out.release()
    vidcap.release()
    vidcnt +=1

        
                
            
##plt.figure(figsize=(10, 10))
##plt.imshow(image)
##show_mask(masks[0], plt.gca())
##show_box(input_box, plt.gca())
##plt.axis('off')
##plt.show()


