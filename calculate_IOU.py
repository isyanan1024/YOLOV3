import os
import numpy as np

from utils import *

def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)


def calculate(path_act,path_pre,labels):
    """
    :param path_act: 人工标注坐标存储路径
    :param path_pre: 预测坐标存储路径
    :return: 平均iou，准确率，召回率
    """

    path_list = os.listdir(path_act)
    path_list.sort()

    mean_iou_list = []
    act_num = 0
    pre_num = 0
    ava_num = 0

    for filename in path_list:
        # print('------------------')
        # print(filename)
        if filename.endswith('.txt'):
            with open(path_act+"/"+filename, "r") as f1:
                act = f1.readlines()
                # print(act)
            if os.path.exists(path_pre+"/"+filename):
                with open(path_pre+"/"+filename, "r") as f2:
                    pre = f2.readlines()
                    # print(pre)
            else:
                print(path_pre+"/"+filename)
                pre =[]
            act_new = []
            pre_new = []

            if len(act) > 0:
                for cell1 in act:
                    cell1 = cell1.replace("\n", "").split(" ")
                    cell1 = [float(i) for i in cell1]
                    X0 = cell1[0]
                    Y0 = cell1[1]
                    X1 = cell1[2]
                    Y1 = cell1[3]

                    box1 = [X0, Y0, X1, Y1]
                    act_new.append(box1)
                    act_len = len(act_new)

            else:
                act_len = 0
                pass
            act_num += act_len

            if len(pre) > 0:
                for cell2 in pre:
                    cell2 = cell2.replace("\n", "").split(" ")
                    cell2_new = [int(i) for i in cell2]
                    X0 = cell2_new[0]
                    Y0 = cell2_new[1]
                    X1 = cell2_new[2]
                    Y1 = cell2_new[3]

                    box2 = [X0, Y0, X1, Y1]
                    pre_new.append(box2)
                pre_len = len(pre_new)
            else:
                pre_len = 0
                pass
            pre_num += pre_len

            try:
                iou_list = []
                while True:
                    max_iou = 0
                    max_i = 0
                    max_j = 0
                    for i, data1 in enumerate(act_new):
                        for j, data2 in enumerate(pre_new):
                            iou = compute_iou(data1, data2)
                            # print(iou)
                            if iou > max_iou: # 0.5以上为有效人形框
                                max_iou = iou
                                max_i = i
                                max_j = j
                    if max_iou > 0.5:
                        iou_list.append(max_iou)
                        act_new.pop(max_i)
                        pre_new.pop(max_j)
                    else:
                        break
                    if len(act_new)*len(pre_new) == 0:
                        break
                mean_iou = sum(iou_list)/len(iou_list)
                mean_iou_list.append(mean_iou)

            except Exception as e:
                iou_list = []
                # print(e)
            # print(iou_list)
            if act_len != pre_len:
                truth_objects = np.loadtxt(os.path.join(path_act, filename))
                predict_objects = np.loadtxt(os.path.join(path_pre, filename))
                save_both_pictures(filename,truth_objects,predict_objects,labels)
            print(filename, act_len, pre_len, len(iou_list))
            ava_num += len(iou_list)

    total_iou = sum(mean_iou_list)/len(mean_iou_list)
    p_value = ava_num/pre_num
    r_value = ava_num/act_num

    return total_iou, p_value, r_value

if __name__ == '__main__':

    info = load_dict()
    labels = [x for x in info['classes'].keys()]
    xml_2_txt('./darknet/dataset/valid.txt', labels)
    labels = load_dict()['classes']
    total_iou, p_value, r_value = calculate("./results_truth", './results_txt',labels)
    print("mean iou:{},p_value:{},r_value:{}".format(round(total_iou,4), round(p_value,4), round(r_value,4)))
    write_dict('iou', round(total_iou, 3))
    write_dict('precision', round(p_value, 3))
    write_dict('recall',round(r_value, 3))