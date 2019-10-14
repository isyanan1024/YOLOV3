from __future__ import print_function
import argparse
import numpy as np
import cv2
import os
import time
# from hat_classify.test import ishat
# from mask_classify.test import ismask


from ctypes import *
import random

VISUALIZATION=True

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


lib = CDLL("./darknet/libdarknet.so", RTLD_GLOBAL)
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

ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE


def nparray_to_image(img):
    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)
    return image


def yolo_detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = nparray_to_image(image)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])

    free_image(im)
    free_detections(dets, num)
    return res

gpu_id = c_int(0)
set_gpu(gpu_id)

yolo_net = load_net(b"./darknet/cfg/yolov3-800.cfg", 
    b"./darknet/backup/yolov3-800_final.weights", 0)
meta = load_meta(b"./darknet/cfg/voc.data")
print('Finished loading model!')

path = '/home/yana/LONGJING/YOLOV3/darknet/dataset/img'
pictures = os.listdir(path)

resumes = []
for picture in pictures:
    if picture.endswith('.jpg'):
        picture_path = os.path.join(path,picture)
        print(picture_path)
        image = cv2.imread(picture_path)
        begin = time.time()
        yolo_dets = yolo_detect(yolo_net, meta, image)
        resume = time.time() - begin
        resumes.append(resume)
        print(1/(sum(resumes)/len(resumes)))

        # visulization
        person_boxes = []

        for i, det in enumerate(yolo_dets):
            flag = True
            box = det[2]
            cx, cy, w, h = np.array(box)
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            # head_image = image[int(y1):int(y2),int(x1):int(x2)]
            # head_image = head_image[...,::-1]
            # hat_result = ishat(head_image)
            # mask_result = ismask(head_image)
            # if hat_result == 0:
            #     cv2.putText(image,'chefhat',(int(x1-5),int(y1-5)),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
            # else:
            #     flag = False
            #     cv2.putText(image,'no_chefhat',(int(x1-5),int(y1-5)),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)

            # if mask_result == 0:
            #     cv2.putText(image,'mask',(int(x1-5),int(y1-30)),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
            # if mask_result == 1:
            #     flag = False
            #     cv2.putText(image,'no_mask',(int(x1-5),int(y1-30)),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
            # if mask_result == 2:
            #     cv2.putText(image,'uncertain_mask',(int(x1-5),int(y1-30)),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)

            if flag:
                cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            else:
                cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
            # with open('txt_pre/'+picture.replace('jpg','txt'),'a+',encoding='utf-8') as f:
            #     f.write(str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n')
        cv2.imwrite(picture,image)




