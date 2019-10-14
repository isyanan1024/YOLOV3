#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 23:18:03 2019

validate the trained model for darknet arch. (recall rate, mean iou and fp)

@author: gongfei
"""

from ctypes import *
import math
import random
import os
from utils import del_mkdirs
import numpy as np
from preprocessing import parse_txt_xm
from tqdm import tqdm
from utils import draw_boxes_PIL,BoundBox_ZK,bbox_iou
import cv2
from PIL import Image
import shutil
from prettytable import PrettyTable
import time


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/yana/LONGJING/YOLOV3/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE



get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

draw_detections = lib.draw_detections
draw_detections.argtypes = [IMAGE, POINTER(DETECTION), c_int, c_float, POINTER(c_char_p), POINTER(POINTER(IMAGE)), c_int]


load_alphabet = lib.load_alphabet
load_alphabet.restype = POINTER(POINTER(IMAGE))

save_image = lib.save_image
save_image.argtypes = [IMAGE, c_char_p]



free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.35 ):
    alphabet = load_alphabet()
    wh = []
    im = load_image(image, 0, 0)
#    h=im.h  #image height
#    w=im.w  #image width
    #print(im.h)
    #print(im.w)
    wh.append(im.w)
    wh.append(im.h)
    
    num = c_int(0)
    pnum = pointer(num)
    
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 1, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], b.x, b.y, b.w, b.h))
    res = sorted(res, key=lambda x: -x[1])
    #draw_detections(im, dets, num, thresh, meta.names, alphabet, meta.classes )
    free_detections(dets, num)

    #save_image(im,outfile)
    free_image(im)

    return res,wh

def return_boxes(boxes_in,labels):   
    boxes_t   = []
    
    for boxi in boxes_in:
        class_id = labels.index(boxi[0].decode())       
        box_x    = boxi[2] 
        box_y    = boxi[3]        
        box_w    = boxi[4] 
        box_h    = boxi[5] 
        
        box_c    = boxi[1]
        
        box_t    = BoundBox_ZK(box_x, box_y, box_w, box_h, class_id, box_c)
        
        boxes_t.append(box_t)
        
    return boxes_t
    
if __name__ == "__main__":
    # set gpu
    gpu_id = c_int(2)
    set_gpu(gpu_id)
    
    '''
    labeltxt_dir = './trainingData/labels/'
    image_dir    = './trainingData/JPEGImages/'
    '''
    
    '''
    labeltxt_dir = './validData/labels/'
    image_dir    = './validData/JPEGImages/'
    '''
    
    labeltxt_dir = '/home/yana/LONGJING/YOLOV3/darknet/dataset/labels/'
    image_dir    = '/home/yana/LONGJING/YOLOV3/darknet/dataset/JPEGImages/'
    
    
    net  = load_net("/home/yana/LONGJING/YOLOV3/darknet/cfg/yolov3-608.cfg".encode('utf-8'), 
                    "/home/yana/LONGJING/YOLOV3/darknet/backup/yolov3-608_final.weights".encode('utf-8'),0)
    
    meta = load_meta("/home/yana/LONGJING/YOLOV3/darknet/cfg/voc.data".encode('utf-8'))
    
    labels_en = ['head']  
    labels_cn = ['头']
    n_class   = len(labels_en)
    
    train_imgs, train_labels = parse_txt_xm(ann_dir = labeltxt_dir, 
                                            img_dir = image_dir,
                                            labels  = labels_en)
    
    
    out_folder   = './all/'
    out_wrong_fd = './wrong/' 
    
    
    '''
    out_folder   = './debug_validSet/class5/all/train/'
    out_wrong_fd = './debug_validSet/class5/wrong/train/'
    '''
    
    
    

    del_mkdirs([out_folder,out_wrong_fd])
    
    fds = [out_wrong_fd + '/' + label for label in labels_en]
    fds.append(out_wrong_fd + '/FN')
    del_mkdirs(fds)
    
    np.set_printoptions(precision = 4)
    
    iou_thresh  = 0.5
    ave_iou     = 0.0
    n_detbox    = 0
    n_totalbox  = 0
    n_fp        = 0.0

    n_detbox_all   = np.zeros(len(labels_en))
    n_totalbox_all = np.zeros(len(labels_en))
    n_fp_all       = np.zeros(len(labels_en))
    ave_iou_all    = np.zeros(len(labels_en))
    
    resume_list = []
    for valid_img in tqdm(train_imgs):
        image_path = valid_img['filename']
        image_name = valid_img['filename'].split('.')[0]
        
        begin = time.time()
        boxes,wh   = detect(net, meta, image_path.encode('utf-8'))
        resume = time.time() - begin
        resume_list.append(resume)
        fps = 1/(sum(resume_list)/len(resume_list))
        print('fps:',fps)

        boxes_draw = return_boxes(boxes,labels_en)
        image      = cv2.imread(image_path)
        image_out  = draw_boxes_PIL(image, boxes_draw, labels_cn, CLASS = n_class, disp_l = 'conf')
        image_out  = image_out[:,:,::-1]
        im         = Image.fromarray(image_out.astype("uint8")) 
        fd,fn      = os.path.split(valid_img['filename'])
        out_fn     = os.path.join(out_folder + fn)
        im.save(out_fn)
        
    
        box_array  = np.array([np.hstack(([labels_en.index(box[0].decode()), box[1],
                                          box[2], box[3], box[4], box[5]])) for box in boxes])
        if box_array.shape[0]>0:
            box_array = box_array[np.argsort(box_array[:,2] + 0.25*box_array[:,3])]
        
        fn_txt,ext    = os.path.splitext(out_fn)
        fout_txt      = fn_txt + '.txt'
        #np.savetxt(fout_txt, box_array,fmt = '%.4f')
        
        
       #------------------------compute iou, recall, FN-----------------------#
        n_obj = 0
        n_det = 0
        b_corr_det = 1
      
        boxes_t = []
        all_objs = valid_img['object']

        fp_label = np.ones(len(boxes_draw),dtype = 'int32')
        fp_curr  = np.zeros(len(labels_en),dtype = 'int32')
        
        for box in all_objs:
            obj_indx,center_x,center_y,center_w,center_h  = int(box[0]),box[1],box[2],box[3],box[4]
            
            n_obj += 1
            n_totalbox_all[obj_indx] += 1
            
            conf        = 1.0
            box_t       = BoundBox_ZK(center_x, center_y, center_w, center_h, obj_indx, conf)
            boxes_t.append(box_t)
            
            max_iou     = -1
            max_iou_pos = -1
            max_lane    = -1

            for id_box,det_box in enumerate(boxes_draw):
                if obj_indx == det_box.get_label():
                    iou = bbox_iou(det_box, box_t)
                    if iou > max_iou:
                        max_iou     = iou
                        max_iou_pos = id_box
            
            if max_iou >= iou_thresh:
                ave_iou                += max_iou
                n_det                  += 1
                ave_iou_all[obj_indx]  += max_iou
                n_detbox_all[obj_indx] += 1
                fp_label[max_iou_pos]  = 0
            else:
                b_corr_det = -1
                print('undetect:',fn)
                
        # add not used det_box for fp
        for id_box, det_box in enumerate(boxes_draw):
            if fp_label[id_box] == 1:
                n_fp_all[det_box.get_label()] += 1
                fp_curr[det_box.get_label()] = 1
        
        n_totalbox += n_obj
        n_detbox   += n_det
        n_fp       += (len(boxes_draw) - n_det) 
        
        if len(boxes_draw) - n_det > 0 and b_corr_det != -1:
            #print('fpdetect ',fn)
            b_corr_det = 0
            
        # save ground-truth
        image_out_t = draw_boxes_PIL(image, boxes_t, labels_cn, CLASS = n_class, disp_l = 'conf')
        image_out_t = image_out_t[:,:,::-1]
        im_t = Image.fromarray(image_out_t.astype("uint8"))
        fn_t,ext =  os.path.splitext(out_fn)
        out_fn1 = fn_t + 't' + ext
        im_t.save(out_fn1)
        
        if b_corr_det != 1:
            if b_corr_det == -1:   #undetect,FN
                fd_tmp = [out_wrong_fd + '/FN']
            elif b_corr_det == 0:  #FP
                fd_tmp = []
                for id_label, label in enumerate(labels_en):
                    if fp_curr[id_label] == 1:
                        fd_tmp += [out_wrong_fd + '/' + label]
            
            im_path,fn  = os.path.split(out_fn)
            im_path,fn1 = os.path.split(out_fn1)
            for fd in fd_tmp:
                fout = fd + '/' + fn
                shutil.copyfile(out_fn, fout)
                fout1 = fd +'/' + fn1
                shutil.copyfile(out_fn1, fout1)
        
        del im
        del image_out
    
    f1 = open('compute_0705_38000_35.txt', 'w')


    table = PrettyTable(["Class", "Number", "Recall%","IOU", "FPs"])
    for ind, label in enumerate(labels_en):
        table.add_row([label, 
                       int(n_totalbox_all[ind]), 
                       np.round(100.*n_detbox_all[ind]/n_totalbox_all[ind],decimals=2),
                       np.round(ave_iou_all[ind]/n_totalbox_all[ind],decimals=3), 
                       int(n_fp_all[ind])]
                     )
        
    f1.close()   

    table.add_row(['total', 
                   int(n_totalbox), 
                   np.round(100.*n_detbox/n_totalbox, decimals=2), 
                   np.round(ave_iou/n_detbox,decimals=3), 
                   int(n_fp)]
                   )
    
    print(table)
    
    data_string = table.get_string()
    
    
    with open('debug_0705_38000_35.txt', 'w') as f:
        f.write(data_string)
    f.close()
    
    
    '''
    with open('./results/debug_training_0702_36000_35.txt', 'w') as f:
        f.write(data_string)
    f.close()
    '''
    
    
    
    
                
