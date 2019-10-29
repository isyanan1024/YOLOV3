# -*- coding: utf-8 -*-
# @Time    : 2019-10-15 16:40
# @Author  : Yan An
# @Contact: an.yan@intellicold.ai

from utils import load_dict

print('generate test cfg')

info = load_dict()

input_size = info['input_size']
train_cfg = './darknet/cfg/yolov3-' + str(input_size) + '.cfg'
test_cfg = './darknet/cfg/yolov3-' + str(input_size) + '_test.cfg'

f = open(test_cfg, 'w', encoding = 'utf-8')

with open(train_cfg, 'r', encoding = 'utf-8') as f2:
	contents = f2.readlines()

	for content in contents:

		if content.startswith('# batch=1'):
			content = 'batch=1' + '\n'
			print('batch=1')

		if content.startswith('# subdivisions=1'):
			content = 'subdivisions=1' + '\n'
			print('subdivisions=1')

		if content.startswith('batch=64'):
			content = '# batch=64' + '\n'
			print('# batch=64')

		if content.startswith('subdivisions=8') or content.startswith('subdivisions=16'):
			content = '# subdivisions=16' + '\n'
			print('# subdivisions=16')

		f.write(content)


