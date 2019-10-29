# -*- coding: utf-8 -*-
# @Time    : 2019-10-19 14:44
# @Author  : Yan An
# @Contact: an.yan@intellicold.ai

import os
import xml.etree.ElementTree as xml_tree

from tqdm import tqdm
from utils import write_dict


path = './darknet/dataset/img'

files = [x for x in os.listdir(path) if x.endswith('xml')]

write_dict('total', len(files))

n_classes = {}
n_classes_index = 0

print('Convert xml to txt')
for file in tqdm(files):
	xml_path = os.path.join(path,file)
	txt_name = xml_path.replace('xml', 'txt')
	if os.path.exists(txt_name):
		continue
	tree=xml_tree.parse(xml_path)
	root=tree.getroot()

	size = root.find('size')
	width = int(size.find('width').text)
	height = int(size.find('height').text)

	f = open(txt_name, 'w',encoding='utf-8')

	objects = root.findall('object')
	for element in objects:

		name = element.find('name').text
		name = 'head'
		if name not in n_classes.keys():
			n_classes[name] = n_classes_index
			n_classes_index += 1

		bndbox = element.find('bndbox')

		xmin = int(float(bndbox.find('xmin').text))
		ymin = int(float(bndbox.find('ymin').text))
		xmax = int(float(bndbox.find('xmax').text))
		ymax = int(float(bndbox.find('ymax').text))

		center_x = round(((xmax - xmin) / 2 + xmin) / width, 6)
		center_y = round(((ymax - ymin)/ 2 + ymin) / height, 6)
		yolo_width = round((xmax - xmin) / width, 6)
		yolo_height = round((ymax - ymin) / height, 6)
		# f.write(str(n_classes[name]) + ' ' + str(center_x) + ' ' + str(center_y) + ' ' + str(yolo_width) + ' ' + str(yolo_height) + '\n')
		f.write(str(0) + ' ' + str(center_x) + ' ' + str(center_y) + ' ' + str(yolo_width) + ' ' + str(yolo_height) + '\n')
	f.close()

f = open('./darknet/cfg/voc.data', 'w', encoding = 'utf-8')
f.write('classes=' + str(len(n_classes.keys())) + '\n')
path = os.path.dirname(os.path.abspath(__file__))
f.write('train = ' + os.path.join(path,'darknet/dataset/train.txt') + '\n')
f.write('valid = ' + os.path.join(path,'darknet/dataset/valid.txt') + '\n')
f.write('names = ' + os.path.join(path,'darknet/data/voc.names') + '\n')
f.write('backup = backup')
f.close()

with open('./darknet/data/voc.names', 'w', encoding = 'utf-8') as f:
	for i in n_classes.keys():
		f.write(i + '\n')

write_dict('classes', n_classes)
write_dict('n_classes', len(n_classes.keys()))