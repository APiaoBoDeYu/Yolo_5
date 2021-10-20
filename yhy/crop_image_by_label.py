#coding=utf-8
# --------------------------------------------------------
# Written by LeiZhang
# @2020.11
#切出小图检查标注质量
# --------------------------------------------------------
import cv2
import os
import  xml.dom.minidom
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import shutil,traceback
path ="./photo/"#"./misinformation/"# "./photo/"
xmlpath ='./xml_ori_bak/'
savepath = "./DRAW/"#"./draw_only_xml/"
from tqdm import tqdm
pbar=tqdm(range(len(os.listdir(path))))
useful_tag = ['BKXQS','XCXJ_XCCJJ_BKXQS_yes']#'JYZ_dhss','JYZ_zb','JYZ_ps'
attr_name_list = ['BKXQS_if']
def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img
# cv2.namedWindow("ss")
# cv2.resizeWindow("ss",width=500,height=500)
num = 1
nameNum = 0
savepicNum = 0
for roots,dirs,files in os.walk(path):
    #print("root is {}".format(roots))
    #print("dirs is {}".format(dirs))
    #print(files)
    for file in files:
        #file = file.decode("utf-8").encode("gbk")
        try:
            if file.split('.')[-1] == "jpg" or file.split('.')[-1] == "JPG":
                # xmlfile = roots +"_XML/"+ file[0:-3] + "xml"
                # xmlfile =xmlpath+ file[0:-3] + "xml"
                pbar.update(1)
                xmlfile = os.path.join(xmlpath, file.split('.')[0] + ".xml")

                if os.path.isfile(roots + "/" + file):
                    img = cv_imread(roots + "/" + file)  # 打开当前路径图像
                    iscontain = False
                    # print("read image")
                    # print(roots +"/"+ file)
                    try:
                        dom = xml.dom.minidom.parse(xmlfile)
                        print(roots + "/" + file)
                        root = dom.documentElement
                        labelList = root.getElementsByTagName('object')
                    except:
                        continue

                        print("Get errors when read image!")
                    for label in labelList:
                        # str_name = str(label.getElementsByTagName('name')[0].firstChild.data)
                        str_name = str(label.getElementsByTagName('name')[0].firstChild.data)
                        for attr_name in attr_name_list:
                            if len(label.getElementsByTagName(attr_name)) > 0:
                                attr_value = str(label.getElementsByTagName(attr_name)[0].firstChild.data)
                                str_name += '_' + attr_value


                        #print(str_name)
                        if not str_name in useful_tag:
                            continue
                        print(str_name)
                        nameNum += 1
                        xmin = int(
                            label.getElementsByTagName('bndbox')[0].getElementsByTagName('xmin')[0].firstChild.data)
                        ymin = int(
                            label.getElementsByTagName('bndbox')[0].getElementsByTagName('ymin')[0].firstChild.data)
                        xmax = int(
                            label.getElementsByTagName('bndbox')[0].getElementsByTagName('xmax')[0].firstChild.data)
                        ymax = int(
                            label.getElementsByTagName('bndbox')[0].getElementsByTagName('ymax')[0].firstChild.data)
                        if not os.path.isdir(savepath + "/" + str_name):
                            os.makedirs(savepath + "/" + str_name)
                        num += 1
                        saveimage = img[ymin:ymax, xmin:xmax]

                        # cv2.imshow('',saveimage)
                        # c=cv2.waitKey(10)
                        cv2.imencode(".%s" % file.split('.')[-1],
                                     saveimage)[1].tofile(
                            savepath + "/" + str_name + "/" + file.split('.')[0] + "_" + str(num) + "_" + str(
                                xmin) + "_" + str(ymin) + ".jpg")
                        savepicNum += 1
                        print("######Get useful name : %s , save small pic : %s"%(nameNum,savepicNum))

        except:
            traceback.print_exc()
            continue


