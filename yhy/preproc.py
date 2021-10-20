# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2019-10-09 15:16:54
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2019-10-09 22:04:00
import xml.etree.ElementTree as ET
import os
import argparse
import sys
from tqdm import tqdm
import cv2
from tqdm import tqdm
import numpy as np

# 小，16个

classes = ["37_NZXJ_YJSX","38_XXXJ","41_NZXJ","44_NZXJ_YYX","NZXJ_yjh",
        "NZXJ_yjb","XCXJ1","XCXJ2","XCXJ3",
        "QTJJ","4_Socket Clevis_W&WS","6_Bolt_U&UJ",
        "ZJGB_1","ZJGB_2","ZJGB_3","Yoke_Plate"]


# 大，3个
region_classes =  ['XCXJ_JJC','LJBW_JYZC1_HENG_JJ', 'LJBW_JYZC1_SHU_JJ', 'LJBW_JYZC2_HENG_JJ', 'LJBW_JYZC2_SHU_JJ',
               'LJBW_TD_HENG_JJ', 'LJBW_TD1_SHU_JJ', 'LJBW_TD2_SHU_JJ', 'XCXJ1_JJ', 'XCXJ3_JJ',
               'XCXJ2_JJ']
new_image_id = 0


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def crop(image_id, raw_path, path):
    '''

    :param image_id: 原文件名字，不带后缀
    :param raw_path: 原始照片和xml存放的目录
    :param path: 存储照片和xml存放的目录
    :return:
    '''
    global new_image_id
    #拿出img对应的xml
    in_file = open('%s/xml/%s.xml' % (raw_path, image_id))
    if not os.path.exists('%s/photo/%s.jpg' % (raw_path, image_id)):
        img = cv2.imdecode(np.fromfile('%s/photo/%s.JPG' % (raw_path, image_id), dtype=np.uint8), -1)
    else:
        img = cv2.imdecode(np.fromfile('%s/photo/%s.jpg' % (raw_path, image_id), dtype=np.uint8), -1)
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    # [xmin, xmax, ymin, ymax]
    luosi = []
    #处理小区域
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        xmlbox = obj.find('bndbox')
        xmin = int(xmlbox.find('xmin').text)
        xmax = int(xmlbox.find('xmax').text)
        ymin = int(xmlbox.find('ymin').text)
        ymax = int(xmlbox.find('ymax').text)
        cls_id = classes.index(cls)
        # print(cls)
        luosi.append([ymin, ymax, xmin, xmax, cls_id])
    # print("luosi is {}!!!!!!!!!".format(len(luosi)))

    #处理大区域
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in region_classes:
            continue
        xmlbox = obj.find('bndbox')
        xmin = int(xmlbox.find('xmin').text)
        xmax = int(xmlbox.find('xmax').text)
        ymin = int(xmlbox.find('ymin').text)
        ymax = int(xmlbox.find('ymax').text)
        w = xmax - xmin
        h = ymax - ymin
        ################add code ######################
        iscontain = False
        for item in luosi:
            if item[0] >= ymin and item[1] <= ymax and item[2] >= xmin and item[3] <= xmax:
                iscontain = True
        if iscontain == False:
            continue
        ################add code ######################
        cropped = img[ymin:ymax, xmin:xmax]
        #图片和要求写下来
        cv2.imencode('.jpg', cropped)[1].tofile("%s/images/%06d.jpg" % (path, new_image_id))
        out_file = open(os.path.join(path, 'labels/%06d.txt' % (new_image_id)), 'w')
        out_file.write("{} {}\n".format(w, h))
        new_image_id += 1
        for item in luosi:
            if item[0] >= ymin and item[1] <= ymax and item[2] >= xmin and item[3] <= xmax:
                b = (item[2] - xmin, item[3] - xmin, item[0] - ymin, item[1] - ymin)
                out_file.write("{} {} {} {} {}\n".format(item[4], b[0], b[1], b[2], b[3]))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='train')
    argparser.add_argument('-trdata', type=str, default='/media/zkzx/9f8c0111-3713-4aa2-ae7d-fd35f3300098/BKXQS')
    args = argparser.parse_args()
    # pwd = os.path.abspath(os.path.dirname(__file__))
    # pwd = os.path.abspath(os.path.dirname(pwd))

    # train_dir = r'/home/igi/media/czf/LJJJ_Fittings/train_step2' #存储路径
    train_dir = r'/home/igi/media/czf/LJJJ_Fittings/val_step2' #存储路径
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    #小相片存储
    if not os.path.exists('%s/images' % train_dir):
        os.makedirs('%s/images' % train_dir)
    #小相片label存储
    if not os.path.exists('%s/labels' % train_dir):
        os.makedirs('%s/labels' % train_dir)

    #原数据位置
    # dir=r'/home/igi/media/czf/LJJJ_Fittings/train'
    dir=r'/home/igi/media/czf/LJJJ_Fittings/val'

    for _,_,files in os.walk(os.path.join(dir,'photo')):
        pbar = tqdm(range(len(files)))
        for file in files:
            pbar.update(1)
            crop(file.split('.')[0], raw_path=dir, path=train_dir)

        f.close()


    sys.stdout.write('Process end with exit code 0')
