import random
import os
import shutil



filepath='/home/igi/media/yhy/ljm/yolov5-develop/train_data/SDNC/labels/train'
distence='/home/igi/media/yhy/ljm/yolov5-develop/train_data/SDNC/'

percent= 0.2  #要转移的概率
for item in os.listdir(filepath):
    if random.random() < percent:
        try:
            label_file=os.path.join(filepath,item)
            shutil.move(label_file,os.path.join(distence,'labels/val'))
        except:
            print('move %s to %s has some wrong' % (label_file, os.path.join(distence,'labels/val')))
            break
        try:
            img_file=os.path.join(distence,'images/train/'+item.split('.')[0]+'.JPG')
            shutil.move(img_file,os.path.join(distence,'images/val'))
        except:
            print('move %s to %s has some wrong'%(img_file,os.path.join(distence,'images/val')))
            break





