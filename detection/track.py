# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from ultralytics import YOLO
import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def get_summary(source_txt, target_txt,  min_track_len=1, min_track_conf=0.0):
    source = {}
    with open(source_txt, 'r') as f:
        for line in f.readlines():
            vid_name, clss, ms, frame, id, _ ,_, _, _, conf = line.split()
            id = int(id)
            if id not in source:
                source[id] = [[ms],[], [], clss]
            source[id][1].append(float(conf))
            source[id][2].append(int(frame))
    result = []
    for id, info in source.items():
        if len(info[1])>=min_track_len:
            fid_min = info[0][0]
            conf = np.mean(info[1])
            if conf>=min_track_conf:
                max_conf = np.max(info[1])
                frame_length = max(info[2])-min(info[2])+1
                result.append([vid_name, info[3], id, fid_min, frame_length, "%.3f"%(len(info[2])/frame_length),"%.6f"%conf,"%.6f"%max_conf])
    with open(target_txt, 'a') as f:
        for line in result:
            f.write(('%s ' * 8 + '\n') % (str(line[0]), str(line[1]), str(line[2]), str(line[3]), str(line[4]), str(line[5]), str(line[6]), str(line[7])))

    """
    summary = []
    for key, value in track_result.items():
        value = np.asarray(value)
        fid_min = int(value[0][1])
        fid_max = int(value[-1][1])
        conf = np.mean(value[:, -1].astype(np.float))
        # if fid_max-fid_min < 5:
        #    continue
        summary.append([key, fid_min, fid_max - fid_min, "%.6f" % conf])
    np.savetxt(os.path.join(FLAGS.output_root, filename + "_summary.csv"), np.asarray(summary), delimiter=",",
               fmt='%s')
    """



def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok, save_img, save_clip, save_bbox, expend_ratio,  min_track_len, min_track_conf = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok, opt.save_img, opt.save_clip, opt.save_bbox, opt.expend_ratio,  opt.min_track_len, opt.min_track_conf
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    device = select_device(opt.device)
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    # Initialize
    
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    #if not evaluate:
    #    if os.path.exists(out):
    #        pass
    #        shutil.rmtree(out)  # delete output folder
    #    os.makedirs(out)  # make new output folder

    # Directories
    #print(Path(project) )
    save_dir = Path(project) #increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    #print(yolo_model)
    model = YOLO(yolo_model).to(device)
    stride = 32
    names = ["other","fish"]
    pt = True
    jit = False
##    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
##    imgsz = check_img_size(imgsz, s=stride)  # check image size


    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    #txt_file_name = source.split('/')[-1].split('.')[0]
   # txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0, 0.0], 0

    tmp_vid = ''
    frame_cnt = 0
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        vid_name = os.path.splitext(os.path.split(path)[1])[0]

        if save_img:
            img_save_folder = os.path.join(str(Path(save_dir)), vid_name, 'images')
            if not os.path.exists(img_save_folder):
                os.makedirs(img_save_folder)
        if save_clip:
            clip_save_folder = os.path.join(str(Path(save_dir)), vid_name, 'clips')
            if not os.path.exists(clip_save_folder):
                os.makedirs(clip_save_folder)

        if save_bbox:
            bbox_save_folder = os.path.join(str(Path(save_dir)), vid_name, 'bboxes')
            if not os.path.exists(bbox_save_folder):
                os.makedirs(bbox_save_folder)

        if tmp_vid != vid_name:
            if tmp_vid != '':
                dest_txt = str(Path(save_dir)) + '/' + tmp_vid + '_summary.txt'
                if not os.path.isfile(str(Path(save_dir)) + '/' + tmp_vid + '.txt'):
                    f = open(str(Path(save_dir)) + '/' + tmp_vid + '.txt', 'w')
                    f.close()
                get_summary(str(Path(save_dir)) + '/' + tmp_vid + '.txt', dest_txt, min_track_len, min_track_conf)



            obj_vid_writer_all = {}
            fps = round(vid_cap.get(cv2.CAP_PROP_FPS))
            deepsort = DeepSort(deep_sort_model,
                                device,
                                max_dist=cfg.DEEPSORT.MAX_DIST,
                                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                                nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                )
            tmp_vid = vid_name
            frame_cnt = 0

        txt_path = str(Path(save_dir)) + '/' + vid_name + '.txt'

        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        frame_cnt += 1
        pred = model.predict(img, conf=opt.conf_thres, verbose=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        #pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, result in enumerate(pred):  # detections per image
            det = result.boxes
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            ids_in_this_frame = []
            deleted = []
            outputs = []
            if det is not None and len(det):
                xywhs = det.xywh.cpu()#xyxy2xywh(det[:, 0:4])
                confs = det.conf.cpu()
                clss = det.cls.cpu()
                # Rescale boxes from img_size to im0 size
                xywhs = scale_coords(
                    img.shape[2:], xywhs, im0.shape).round()


                
                # Print results
                for c in clss.unique():
                    n = (clss == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                

                # pass detections to deepsort
                t4 = time_sync()
                outputs, deleted = deepsort.update(xywhs, confs, clss, im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        ids_in_this_frame.append(id)

                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            sec = int((frame_cnt+1)/fps)%60
                            minute = int((frame_cnt + 1) / fps) // 60
                            hour = 0
                            if minute>=60:
                                hour = minute//60
                                minute = minute%60

                            with open(txt_path, 'a') as f:
                                f.write(('%s ' * 10 + '\n') % (vid_name, names[int(c)], str(hour).rjust(2,'0')+":"+str(minute).rjust(2,'0')+":"+str(sec).rjust(2,'0'), frame_cnt+1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, conf.cpu().numpy()))
                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), Tracking:({t5 - t4:.3f}s)')
            else:
                deepsort.increment_ages()
                # LOGGER.info('No detections')

            # Stream results
            t6 = time_sync()
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
            if save_img and len(outputs):
                sec = int((frame_cnt + 1) / fps) % 60
                minute = int((frame_cnt + 1) / fps) // 60
                hour = 0
                if minute >= 60:
                    hour = minute // 60
                    minute = minute % 60
                img_name = str(hour).rjust(2,'0')+"_"+str(minute).rjust(2,'0')+"_"+str(sec).rjust(2,'0')+'.png'
                img_path = img_save_folder + '/' +img_name
                cv2.imwrite(img_path, im0)

            if save_bbox and len(outputs):
                sec = int((frame_cnt + 1) / fps) % 60
                minute = int((frame_cnt + 1) / fps) // 60
                hour = 0
                if minute >= 60:
                    hour = minute // 60
                    minute = minute % 60

                for j, (output, conf) in enumerate(zip(outputs, confs)):
                    h, w, _ = im0.shape
                    x1, y1, x2, y2, id, clss = output[0:6]
                    bbox_name = names[int(clss)]
                    bbox_h = y2-y1
                    bbox_w = x2-x1
                    conf = "%.4f" % conf
                    img_name = bbox_name+ "_object_"+str(id)+"_"+str(hour).rjust(2, '0') + "_" + str(minute).rjust(2, '0') + "_" + str(sec).rjust(2,'0') + '_'+ conf[2:]+'.png'
                    crop_img = im0[max(0, int(y1-bbox_h*expend_ratio)):min(h,int(y2+bbox_h*expend_ratio)), max(0, int(x1-bbox_w*expend_ratio)):min(w,int(x2+bbox_w*expend_ratio))]
                    bbox_path = bbox_save_folder + '/' + img_name
                    cv2.imwrite(bbox_path, crop_img)

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

            if save_clip and len(outputs):
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                for obj_id in ids_in_this_frame:
                    if obj_id not in obj_vid_writer_all.keys():

                        sec = int((frame_cnt + 1) / fps) % 60
                        minute = int((frame_cnt + 1) / fps) // 60
                        hour = 0
                        if minute >= 60:
                            hour = minute // 60
                            minute = minute % 60
                        start_time = str(hour).rjust(2, '0') + "_" + str(minute).rjust(2, '0') + "_" + str(sec).rjust(2,'0')
                        #obj_clip_path = os.path.join(clip_save_folder, "object_"+str(obj_id)+'_'+start_time+'.mp4')
                        obj_clip_path = clip_save_folder +'/'+ "object_"+str(obj_id)+'_'+start_time+'.mp4'
                        obj_vid_writer_all[obj_id] = cv2.VideoWriter(obj_clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    obj_vid_writer_all[obj_id].write(im0)
                for obj_id in deleted:
                    obj_vid_writer_all[obj_id].release()
            t7 = time_sync()
            dt[4] += t7 - t6
    if tmp_vid != '':
        dest_txt = str(Path(save_dir)) + '/' + tmp_vid + '_summary.txt'
        if not os.path.isfile(str(Path(save_dir)) + '/' + tmp_vid + '.txt'):
            f = open(os.path.isfile(str(Path(save_dir)) + '/' + tmp_vid + '.txt'), 'w')
            f.close()
        get_summary(str(Path(save_dir)) + '/' + tmp_vid + '.txt', dest_txt)
        for objid, writer in obj_vid_writer_all.items():
            writer.release()
        frame_cnt = 0

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.3fms pre-process, %.3fms inference, %.3fms NMS, %.3fms deep sort update, %.3fms post-process\
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='../chuteConfig.yml',
                        help='path to configuration file')
    parser.add_argument('--yolo_model', nargs='+', type=str, default='rockfish_yolov5x_tune.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='./test_videos', help='source')  #file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1088], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_false', help='save MOT compliant results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=100, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'inference/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-img', action='store_true', help='save tracks in images')
    parser.add_argument('--save-bbox', action='store_true', help='save tracks in bbox+small context images')
    parser.add_argument('--save-clip', action='store_true', help='save tracks in video clips')
    parser.add_argument('--expend-ratio', type=float, default=0.2, help='expend bbox and save')
    parser.add_argument('--min-track-len', type=int, default=1, help='min duration of track (number of frames, default is 1)')
    parser.add_argument('--min-track-conf', type=float, default=0.0,help='min average confidence of track (default is 0.0)')
    opt = parser.parse_args()

    print(opt.config_path)
    with open(opt.config_path, 'r') as file:
        config = yaml.safe_load(file)

    opt.yolo_model = config["det_model_path"]
    
    opt.save_vid = config["save_vid"]
    print(opt.conf_thres)
    opt.conf_thres = config["det_thresh"]
    #opt.project = "../"+config["output_dir"]+"/detection"
    opt.name = "detection"
    #opt.source = "../"+config["output_dir"]+"/undistorted"

    if(not os.path.isabs(config["output_dir"])):
        opt.project = "../"+config["output_dir"]+"/detection"
        opt.source = "../"+config["output_dir"]+"/undistorted"
    else:
        opt.project = config["output_dir"]+"/detection"
        opt.source = config["output_dir"]+"/undistorted"

    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
