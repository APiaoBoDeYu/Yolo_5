#*-*coding=utf-8-*

import cv2
import os
import  xml.dom.minidom
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import shutil
if False:
    path ='/home/igi/media/data/ddxljfs/肇庆二期_导地线连接方式_训练集/photo/'#"./misinformation/"# "./photo/"
    xmlpath = '/home/igi/media/data/ddxljfs/肇庆二期_导地线连接方式_训练集/xml/'
    savepath = "/home/igi/media/data/ddxljfs/肇庆二期_导地线连接方式_训练集/check/"#"./draw_only_xml/"

    # 类别'DDXLJ','DXC_ljfs','GDC_ljfs'
    useful_tag = ['DDXLJ','DXC_ljfs','GDC_ljfs']
    # tag_color={'XCXJ_JJC':(255,0,0),'XCXJ':(0,255,0),'44_NZXJ_YYX':(160,32,240),'41_NZXJ':(255,255,0),'XCXJ_XS':(0,255,255)}
    tag_color={'DDXLJ':(255,0,0),'DXC_ljfs':(0,255,0),'GDC_ljfs':(0,0,255)}
    #蓝(255,0,0)，绿（0,255,0),红(0,0,255)，青(255,255,0)

if False:
    path ='/home/igi/media/zyf/yolo_输电鸟巢2/images/'#"./misinformation/"# "./photo/"
    xmlpath = '/home/igi/media/zyf/yolo_输电鸟巢2/images/'
    savepath = "/home/igi/media/zyf/yolo_输电鸟巢2/images/"#"./draw_only_xml/"

    # 类别'DDXLJ','DXC_ljfs','GDC_ljfs'
    useful_tag = ['GTYW_QTYW','GTYW_bird_nest','GTYW_ZLK','JGLSJJ']
    # tag_color={'XCXJ_JJC':(255,0,0),'XCXJ':(0,255,0),'44_NZXJ_YYX':(160,32,240),'41_NZXJ':(255,255,0),'XCXJ_XS':(0,255,255)}
    tag_color={'GTYW_QTYW':(255,0,0),'GTYW_bird_nest':(0,255,0),'GTYW_ZLK':(0,0,255),'JGLSJJ':(255,255,0)}
    #蓝(255,0,0)，绿（0,255,0),红(0,0,255)，青(255,255,0)

if True:
    path ='/home/igi/media/yhy/ljm/yolov5-develop/runs/detect/exp7/'#图像路径
    xmlpath = '/home/igi/media/czf/输电比赛/输电鸟巢/杆塔异物_数据备份_210521/xml/'
    savepath = "/home/igi/media/yhy/ljm/yolov5-develop/runs/detect/exp7/"#存储路径

    useful_tag = ['GTYW_bird_nest']
    tag_color={'GTYW_bird_nest':(255,0,0)}
    #蓝(255,0,0)，绿（0,255,0),红(0,0,255)，青(255,255,0)

def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img

def draw_xml():

    for roots,dirs,files in os.walk(path):
        print("root is {}".format(roots))
        #print("dirs is {}".format(dirs))
        #print(files)
        file_i=0
        for file in files:
            #file = file.decode("utf-8").encode("gbk")
            #print(file)
            if file[-3:] == "jpg" or file[-3:] == "JPG" :
                #xmlfile = roots +"_XML/"+ file[0:-3] + "xml"
                xmlfile =xmlpath+ file[0:-3] + "xml"

                if os.path.isfile(roots +"/"+ file):
                    img = cv_imread(roots +"/"+ file)  # 打开当前路径图像
                    iscontain = False
                    print("read image")
                    try:
                        dom = xml.dom.minidom.parse(xmlfile)
                        print(roots+"/" + file)
                        root = dom.documentElement
                        labelList = root.getElementsByTagName('object')
                    except:
                        continue
                    # NZXJ=[]
                    for label in labelList:
                        #str_name = str(label.getElementsByTagName('name')[0].firstChild.data)
                        str_name = str(label.getElementsByTagName('name')[0].firstChild.data)

                        # #临时
                        # if '44_NZXJ_YYX' ==  str_name:
                        #     NZXJ.append(str_name)
                        # else:
                        #     continue

                        if not str_name in useful_tag:
                            #iscontain = True
                            continue
                        print(str_name)
                        iscontain = True
                        xmin = int(label.getElementsByTagName('bndbox')[0].getElementsByTagName('xmin')[0].firstChild.data)
                        ymin = int(label.getElementsByTagName('bndbox')[0].getElementsByTagName('ymin')[0].firstChild.data)
                        xmax = int(label.getElementsByTagName('bndbox')[0].getElementsByTagName('xmax')[0].firstChild.data)
                        ymax = int(label.getElementsByTagName('bndbox')[0].getElementsByTagName('ymax')[0].firstChild.data)
                        color=tag_color[str_name]
                        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),color,2)
                        cv2.putText(img, str_name, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 2, color, 2)
                        print("draw")
                        #cv2.putText(img, str_name, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255),2)

                    # #用以保存特殊图片
                    # if len(NZXJ):
                    #     file_i+=1
                    cv2.imencode(file[-4:], img)[1].tofile(savepath + file)
                        # print(file_i)

if __name__ == '__main__':
    draw_xml()




