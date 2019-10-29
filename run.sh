#/bin/bash

# #删除已经创建的txt文件
rm -rf ./darknet/dataset/img/*.txt

# #将xml格式数据格式转换成YOLOV3需要的格式
python xml2txt.py

# # #按照7:2:1的比例生成train和valid数据
python split_data.py

# # #聚类anchor框--input_size是32的倍数，图片细节越多，对应精度相对越高，但是速度会下降 --n_clusters聚类的个数
python anchors_kmeans.py --input_size 608 --n_clusters 6

# # #修改参数
python modify_parameters.py

#重新编译
cd darknet
make

# #拷贝libdarknet.so到上级目录
cp libdarknet.so ../libdarknet.so

#开始训练
cd ..
python start_train.py

# 生成测试需要的cfg文件
python generate_test_cfg.py

#删除存放结果的文件夹
rm -rf results/* results_truth/* results_txt/*

# 生成图片和txt结果
python detection.py --gpu_id 0

# 评估模型
python calculate_IOU.py