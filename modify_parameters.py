# -*- coding: utf-8 -*-
# @Time    : 2019-10-16 17:22
# @Author  : Yan An
# @Contact: an.yan@intellicold.ai

import os

from utils import load_dict,write_dict


print('modify parameters...')
info = load_dict()
input_size = info['input_size']
train_num = int(info['train'])

# 没张图过273便，和作者训练的对应
max_batches = int(train_num * 273 / 64)
# 到这一step调整学习旅
step1 = int(max_batches * 0.8)
step2 = int(max_batches * 0.9)

n_clusters = info['n_clusters']
mask_index = 0

n_classes = info['n_classes']

anchors = info['anchors']

f2 = open('./darknet/cfg/yolov3-' + str(input_size) + '.cfg', 'w', encoding = 'utf-8')

with open('./darknet/cfg/yolov3.cfg', 'r', encoding = 'utf-8') as f:
	contents = f.readlines()

	for content in contents:

		if content.startswith('subdivisions'):
			if input_size == 416:
				content = 'subdivisions=8' + '\n'
				print(content.replace('\n',''))
			elif input_size == 608:
				content = 'subdivisions=16' + '\n'
				print(content.replace('\n',''))
			else:
				print(content.replace('\n',''))
				content = 'subdivisions=32' + '\n'

		if content.startswith('width'):
			content = 'width=' + str(input_size) + '\n'
			print(content.replace('\n',''))

		if content.startswith('height'):
			content = 'height=' + str(input_size) + '\n'
			print(content.replace('\n',''))

		if content.startswith('max_batches='):
			content = 'max_batches=' + str(max_batches) + '\n'
			print(content.replace('\n',''))
			write_dict('max_batches', max_batches)

		if content.startswith('steps'):
			content = 'steps=' + str(step1) + ',' + str(step2) +'\n'
			print(content.replace('\n',''))
		#等好之前有个空格，和不用修改的filters区分开来
		if content.startswith('filters ='):
			content = 'filters =' + str(int(n_clusters / 3 * (n_classes + 5))) +'\n'
			print(content.replace('\n',''))

		if content.startswith('mask'):
			if n_clusters == 6:
				content = 'mask=' + str(n_clusters - mask_index -2) + ',' + str(n_clusters - mask_index - 1) +'\n'
				print(content.replace('\n',''))
			if n_clusters == 9:
				content = 'mask=' + str(n_clusters - mask_index -3) + ',' + str(n_clusters - mask_index - 2) + ',' + str(n_clusters - mask_index -1) +'\n'
				print(content.replace('\n',''))
			mask_index += int(n_clusters / 3)

		if content.startswith('anchors'):
			content = 'anchors=' + anchors +'\n'
			print(content.replace('\n',''))

		if content.startswith('classes'):
			content = 'classes=' + str(n_classes) +'\n'
			print(content.replace('\n',''))

		if content.startswith('num'):
			content = 'num=' + str(n_clusters) +'\n'
			print(content.replace('\n',''))

		f2.write(content)

f2.close()

f3 = open('./darknet/examples/detector.backup', 'a+', encoding = 'utf-8')
with open('./darknet/examples/detector.c', 'r', encoding = 'utf-8') as f:
	contents = f.readlines()

	for content in contents:

		if content.startswith('            int dim = '):
			content = '            int dim = (rand() % 10 + ' + str(int((input_size / 32) - 9)) + ') * 32;' + '\n'
			print(content.replace('\n','').lstrip())

		if content.startswith('            if (get_current_batch(net)+200 > net->max_batches) dim'):
			content = '            if (get_current_batch(net)+200 > net->max_batches) dim = ' + str(input_size) + ';' + '\n'
			print(content.replace('\n','').lstrip())

		f3.write(content)
f3.close()

os.remove('./darknet/examples/detector.c')
os.rename('./darknet/examples/detector.backup','./darknet/examples/detector.c')
