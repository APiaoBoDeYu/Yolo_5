#coding=utf-8
# --------------------------------------------------------
# Written by LeiZhang
# @2020.11
#切出小图检查标注质量
# --------------------------------------------------------
import codecs

import cv2
import os
import  xml.dom.minidom
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import shutil,traceback
path = "./DRAW/"#"./draw_only_xml/"
from tqdm import tqdm
import glob
path_file_number=glob.glob('D:/case/test/testcase/checkdata/*.py')
pbar=tqdm(range(100000))
useful_tag = ['BKXQS']#'JYZ_dhss','JYZ_zb','JYZ_ps'
path_rgb='./byRBG'
path_per = './byPERCENT'
if not os.path.isdir(path_rgb):
    os.makedirs(path_rgb)
if not os.path.isdir(path_per):
    os.makedirs(path_per)
def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img
num = 1
proportion = 0.1
def getSumPixel(statisticRegion):
    bgrNum = [0, 0, 0]
    sHeight, sWidth, s_ = statisticRegion.shape
    totalNum = sHeight * sWidth

    if totalNum == 0:
        return 0,0,0
    for i in range(sHeight):
        for j in range(sWidth):
            bgrNum[0]  += statisticRegion[i,j][0]
            bgrNum[1] += statisticRegion[i, j][1]
            bgrNum[2] += statisticRegion[i, j][2]
    return bgrNum[0]/totalNum,bgrNum[1]/totalNum,bgrNum[2]/totalNum

for roots,dirs,files in os.walk(path):
    for file in files:
        #file = file.decode("utf-8").encode("gbk")
        try:
            if file.split('.')[-1] == "jpg" or file.split('.')[-1] == "JPG":
                pbar.update(1)
                img = cv_imread(roots + "/" + file)  # 打开当前路径图像
                height,width,_=img.shape
                xmin = int(width  * (1-proportion)/2.0)
                xmax = int(width  * ((1-proportion)/2.0 + proportion))
                ymin = int(height * (1-proportion)/2.0)
                ymax = int(height * ((1-proportion)/2.0 + proportion))
                statisticRegion = img[ymin:ymax, xmin:xmax]
                b,g,r = getSumPixel(statisticRegion)
                brightness = r*0.299+g*0.587+b*0.114
                savedirnane = os.path.join(path_rgb,str(int(brightness)))
                if not os.path.isdir(savedirnane):
                    os.makedirs(savedirnane)
                shutil.copy(roots + "/" + file,savedirnane + '/' +file )
                print(b,g,r)
        except:
            traceback.print_exc()
            continue

