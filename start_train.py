import os

from utils import load_dict


info = load_dict()
input_size = info['input_size']
os.chdir('darknet')
os.system('echo $(pwd)')

if not os.path.exists('./backup/yolov3-' + str(input_size) + '.backup'):
	os.system('./darknet detector train cfg/voc.data cfg/yolov3-' + str(input_size) + '.cfg darknet53.conv.74 -gpus 0,1')
else:
	os.system('./darknet detector train cfg/voc.data cfg/yolov3-' + str(input_size) + '.cfg backup/yolov3-' + str(input_size) + '.backup 0,1')
