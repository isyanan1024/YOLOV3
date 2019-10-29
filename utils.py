# -*- coding: utf-8 -*-
# @Time    : 2019-10-20 10:56
# @Author  : Yan An
# @Contact: an.yan@intellicold.ai

import os
import cv2
import copy
import json
import numpy as np
import xml.etree.ElementTree as xml_tree

from tqdm import tqdm


def write_dict(key, value):
    if not os.path.exists('info.txt'):
        f = open('info.txt', 'w', encoding = 'utf-8')
        info = {}
        info[key] = value
        f.write(json.dumps(info,indent=2, sort_keys=False))
        f.close()
    else:
        f = open('info.txt', 'r', encoding = 'utf-8')
        info = json.load(f)
        info[key] = value
        f.close()
        f = open('info.txt', 'w', encoding = 'utf-8')
        f.write(json.dumps(info, indent=2,sort_keys=False))
        f.close()

def load_dict():
    with open('info.txt', 'r', encoding = 'utf-8') as f:

        info = json.load(f)

        return info

def get_gpu_info(gpu_id):
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    GPU_type = pynvml.nvmlDeviceGetName(handle).decode()
    GPU_used = meminfo.used / 1024**2

    return GPU_type,GPU_used

def xml_2_txt(valid_txt_path, labels):
    print('convert xml to txt...')
    f = open(valid_txt_path, 'r', encoding = 'utf-8')
    valid_images_path = f.readlines()
    f.close()

    for valid_image_path in tqdm(valid_images_path):
    
        xml_path = valid_image_path.replace('\n','').replace('jpg','xml')

        txt_name = xml_path.split('/')[-1].split('.')[0] + '.txt'
        f = open('results_truth/' + txt_name, 'w',encoding='utf-8')

        tree=xml_tree.parse(xml_path)
        root=tree.getroot()

        objects = root.findall('object')
        for element in objects:

            name = element.find('name').text
            name = 'head'
            index = labels.index(name)

            bndbox = element.find('bndbox')

            xmin = int(float(bndbox.find('xmin').text) + 0.5)
            ymin = int(float(bndbox.find('ymin').text) + 0.5)
            xmax = int(float(bndbox.find('xmax').text) + 0.5)
            ymax = int(float(bndbox.find('ymax').text) + 0.5)

            f.write(str(index) + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax) + '\n')
    f.close()

def expandDims(object):
    return np.expand_dims(object, axis = 0)

def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3

def bbox_iou(box1, box2):
    x1_min, x1_max, y1_min, y1_max = box1[1], box1[3], box1[2], box1[4]

    x2_min, x2_max, y2_min, y2_max = box2[1], box2[3], box2[2], box2[4]

    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])

    intersect = intersect_w * intersect_h

    union = (x1_max - x1_min) * (y1_max - y1_min) + (x2_max - x2_min) * (y2_max - y2_min) - intersect

    return float(intersect) / union

def save_both_pictures(txt, truth_objects, predict_objects, labels):

    img = cv2.imread(os.path.join('./darknet/dataset/img/' + txt.replace('txt', 'jpg')))
    copy_img = copy.deepcopy(img)

    if len(truth_objects.shape) == 1:
          truth_objects = expandDims(truth_objects)
    for truth_object in truth_objects:
        xmin, ymin, xmax, ymax = int(truth_object[1]), int(truth_object[2]), int(truth_object[3]), int(truth_object[4]) 
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # cv2.putText(img,labels[truth_object[0]],(int(x1), int(y1) - 10),cv2.FONT_HERSHEY_SIMPLEX,2,(0, 0, 255), 2)

    if len(predict_objects) != 0:
        if len(predict_objects.shape) == 1:
          predict_objects = expandDims(predict_objects)
        for predict_object in predict_objects:
            xmin, ymin, xmax, ymax = int(predict_object[1]), int(predict_object[2]), int(predict_object[3]), int(predict_object[4])
            cv2.rectangle(copy_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            # cv2.putText(copy_img,labels[predict_object[0]],(int(x1), int(y1) - 10),cv2.FONT_HERSHEY_SIMPLEX,2,(0, 0, 255), 2)

    save_img = np.hstack([img, copy_img])
    cv2.imwrite('./wrong/' + txt.replace('txt', 'jpg'), save_img)




