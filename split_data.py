import os
import random

from tqdm import tqdm
from utils import write_dict

path = './darknet/dataset/img'

files = os.listdir(path)
pictures = [x for x in files if x.endswith('jpg')]

train_samples = random.sample(pictures,int(0.7*len(pictures)))

path = os.path.dirname(os.path.abspath(__file__))
f_train = open('./darknet/dataset/' + 'train.txt','w',encoding='utf-8')
f_valid = open('./darknet/dataset/' + 'valid.txt','w',encoding='utf-8')

print('generate train txt')
for train_sample in tqdm(train_samples):
	f_train.write(os.path.join(path,'darknet/dataset/img/') + train_sample + '\n')


valid_samples = [x for x in pictures if x not in train_samples]

print('generate valid txt')
for valid_sample in tqdm(valid_samples):
	f_valid.write(os.path.join(path,'darknet/dataset/img/') + valid_sample + '\n')

write_dict('train', len(train_samples))
write_dict('valid', len(valid_samples))