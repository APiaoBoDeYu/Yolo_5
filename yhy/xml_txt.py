import os
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm
import random


step='xdqs'
data='val'
#第一步模型
if step ==1:
    if data == 'train':
        dirpath = r'/home/igi/media/yhy/casecade/train_data/xml'  # 原来存放xml文件的目录
        newdir = r'../xiushi/labels/train'  # 修改label后形成的txt目录
        photodir = r'/home/igi/media/yhy/casecade/train_data/photo/'
        photonewdir = r'../xiushi/images/train/'
    elif data =='val':
        dirpath = r'/home/igi/media/yhy/jinjuxiushi/test_dataset2/xml'
        newdir = r'../xiushi/labels/val'
        photodir = r'/home/igi/media/yhy/jinjuxiushi/test_dataset2/photo/'
        photonewdir = r'../xiushi/images/val/'
    else:
        pass
# 第二步模型
elif step==2:
    if data == 'train':
        dirpath = r'/home/igi/media/yhy/casecade/train_data/step2_labels'  # 原来存放xml文件的目录
        newdir = r'../xiushi_true/labels/train'  # 修改label后形成的txt目录
        # photodir = r'/home/igi/media/yhy/casecade/train_data/photo/'
        # photonewdir = r'../xiushi/images/train/'
    elif data =='val':
        pass
elif step=='sdnc':
    dirpath = r'/home/igi/media/czf/输电比赛/输电鸟巢/杆塔异物_数据备份_210521/xml/'  # 原来存放xml文件的目录
    newdir = r'/home/igi/media/yhy/ljm/yolov5-develop/train_data/SDNC/labels/train/'  # 修改label后形成的txt目录
    photodir = r'/home/igi/media/czf/输电比赛/输电鸟巢/杆塔异物_数据备份_210521/photo/'
    photonewdir = r'/home/igi/media/yhy/ljm/yolov5-develop/train_data/SDNC/images/train/'
elif step=='xdqs':
    if data=='train':
        dirpath = r'/home/igi/media/czf/LJJJ_Fittings/train/xml/'  # 原来存放xml文件的目录
        newdir = r'/home/igi/media/yhy/ljm/yolov5-develop/train_data/LJJJ_Fittings/labels/train/'  # 修改label后形成的txt目录
        photodir = r'/home/igi/media/czf/LJJJ_Fittings/train/photo/'
        photonewdir = r'/home/igi/media/czf/LJJJ_Fittings/train/temp/'
    else:
        dirpath = r'/home/igi/media/czf/LJJJ_Fittings/val/xml/'  # 原来存放xml文件的目录
        newdir = r'/home/igi/media/yhy/ljm/yolov5-develop/train_data/LJJJ_Fittings/labels/val/'  # 修改label后形成的txt目录
        photodir = r'/home/igi/media/czf/LJJJ_Fittings/val/photo/'
        photonewdir = r'/home/igi/media/czf/LJJJ_Fittings/val/temp/'

else:
    pass
if not os.path.exists(newdir):
    os.makedirs(newdir)
if not os.path.exists(photonewdir):
    os.makedirs(photonewdir)


def xml_txt_step2(dirpath):
    pbar = tqdm(range(len(os.listdir(dirpath))))
    # 逐个拿出文件
    for fp in os.listdir(dirpath):
        lines=open(os.path.join(dirpath,fp)).readlines()
        width,height=[int(i) for i in lines[0].split(' ')]
        for line in lines[1:]:
            label,xmin, xmax,ymin, ymax=[int(i) for i in line.split(' ')]
            x_center = (xmin + xmax) / (2 * width)
            y_center = (ymin + ymax) / (2 * height)
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            with open(os.path.join(newdir, fp.split('.')[0] + '.txt'), 'a+') as f:
                # print(fp)
                f.write(' '.join([str(label), str(x_center), str(y_center), str(w), str(h) + '\n']))
                temp = True
        pbar.update(1)
def xml_txt(dirpath,newdir,classes_originID,check_class):
    pbar = tqdm(range(len(os.listdir(dirpath))))
    for fp in os.listdir(dirpath):

        root = ET.parse(os.path.join(dirpath, fp)).getroot()

        xmin, ymin, xmax, ymax = 0, 0, 0, 0
        sz = root.find('size')

        width = float(sz[0].text)
        height = float(sz[1].text)
        filename = root.find('filename').text
        temp = False
        for child in root.findall('object'):  # 找到图片中的所有框
            # print(child.find('name').text)

            sub = child.find('bndbox')  # 找到框的标注值并进行读取
            label = child.find('name').text

            classes_originID = classes_originID

            if label  in check_class:
                try:
                    label = classes_originID[label]
                    xmin = max(float(sub[0].text), 0)
                    ymin = max(float(sub[1].text), 0)
                    xmax = float(sub[2].text)
                    ymax = float(sub[3].text)
                except:
                    print('坐标缺失')
                    continue
                if xmax < 0 or ymax < 0:
                    continue
                try:  # 转换成yolov5的标签格式，需要归一化到（0-1）的范围内
                    x_center = (xmin + xmax) / (2 * width)
                    y_center = (ymin + ymax) / (2 * height)
                    w = (xmax - xmin) / width
                    h = (ymax - ymin) / height
                except ZeroDivisionError:
                    print(filename, '的 width有问题')

                with open(os.path.join(newdir, fp.split('.')[0] + '.txt'), 'a+') as f:
                    # print(fp)
                    f.write(' '.join([str(label), str(x_center), str(y_center), str(w), str(h) + '\n']))
                    temp = True
        pbar.update(1)
        #如果图片没有响应的label转移图片
        if not temp:
            if os.path.exists(photodir + fp.split('.')[0]+'.JPG') and os.path.exists(photonewdir + fp.split('.')[0] + '.JPG') == False:
                shutil.move(photodir + fp.split('.')[0]+'.JPG',photonewdir + fp.split('.')[0] + '.JPG')
            elif os.path.exists(photodir + fp.split('.')[0]+'.jpg') and os.path.exists(photonewdir + fp.split('.')[0] + '.JPG') == False:
                shutil.move(photodir + fp.split('.')[0]+'.jpg',photonewdir + fp.split('.')[0] + '.JPG')

        #该模块用于保存图片
        # if os.path.exists(photonewdir + fp.split('.')[0] + '.JPG'):
        #     continue
        # if temp == True:
        #
            # if os.path.exists(photodir + fp.split('.')[0]+'.JPG') and os.path.exists(photonewdir + fp.split('.')[0] + '.JPG') == False:
            #     shutil.copy(photodir + fp.split('.')[0]+'.JPG',
            #                 photonewdir + fp.split('.')[0] + '.JPG')
            #     # shutil.move(photodir + fp.split('.')[0]+'.JPG',photonewdir + fp.split('.')[0] + '.JPG')
            # elif os.path.exists(photodir + fp.split('.')[0]+'.jpg') and os.path.exists(photonewdir + fp.split('.')[0] + '.JPG') == False:
            #     shutil.copy(photodir + fp.split('.')[0] + '.jpg',
            #                 photonewdir + fp.split('.')[0] + '.JPG')
            #     # shutil.move(photodir + fp.split('.')[0]+'.jpg',photonewdir + fp.split('.')[0] + '.JPG')
        #
        #     else:
        #         print('图片不存在')
        # pbar.update(1)

    print('ok')

if __name__ == '__main__':
    # classes_originID = {'DXC_ljfs_one':0,'DXC_ljfs_two':1,'GDC_ljfs_one':2,'GDC_ljfs_two':3}
    # check_class=['DXC_ljfs','GDC_ljfs']
    classes_originID = {"37_NZXJ_YJSX":0,"38_XXXJ":1,"41_NZXJ":2,"44_NZXJ_YYX":3,"NZXJ_yjh":4,
               "NZXJ_yjb":5,"XCXJ1":6,"XCXJ2":7,"XCXJ3":8,
               "QTJJ":9,"4_Socket Clevis_W&WS":10,"6_Bolt_U&UJ":11,
               "ZJGB_1":12,"ZJGB_2":13,"ZJGB_3":14,"Yoke_Plate":15}
    check_class=["37_NZXJ_YJSX","38_XXXJ","41_NZXJ","44_NZXJ_YYX","NZXJ_yjh",
               "NZXJ_yjb","XCXJ1","XCXJ2","XCXJ3",
               "QTJJ","4_Socket Clevis_W&WS","6_Bolt_U&UJ",
               "ZJGB_1","ZJGB_2","ZJGB_3","Yoke_Plate"]
    # The_class={"37_NZXJ_YJSX","38_XXXJ","41_NZXJ","44_NZXJ_YYX","NZXJ_yjh",
    #            "NZXJ_yjb","44_NZXJ_YYX","XCXJ1","XCXJ2","XCXJ3"
    #            "QTJJ","4_Socket Clevis_W&WS","6_Bolt_U&UJ",
    #            "ZJGB_1","ZJGB_2","ZJGB_3","Yoke_Plate"}
    # dirpath = r'/home/igi/media/data/ddxljfs/肇庆二期_导地线连接方式_测试集/xml'  # 原来存放xml文件的目录
    # newdir = r'../train_data/DDXLJ/labels/val/'  # 修改label后形成的txt目录

    if not os.path.exists(newdir):
        os.makedirs(newdir)


    xml_txt(dirpath,newdir,classes_originID,check_class)
