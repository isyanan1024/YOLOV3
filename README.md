# YOLOV3

## 训练

运行bash run.sh自动完成数据分类、训练和评估，结果在info.txt中。需要先下载darknet53.conv.74（链接:https://pan.baidu.com/s/1wQicuVidzUn0-3I--oEzxw  密码:1kot）在darknet文件夹下。

info.txt展示：

{
  "total": 24267,
  "classes": {
    "head": 0
  },
  "n_classes": 1,
  "train": 16986,
  "valid": 7281,
  "input_size": 416,
  "n_clusters": 6,
  "anchors": " 12,24, 16,35, 20,47, 28,64, 42,89, 65,140",
  "max_batches": 72455,
  "FPS": 25,
  "GPU_type: ": "GeForce RTX 2080 Ti",
  "GPU_used: ": "1907M",
  "iou": 0.877,
  "precision": 0.986,
  "recall": 0.983
}