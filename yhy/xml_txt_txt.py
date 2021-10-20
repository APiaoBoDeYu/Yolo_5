#此文件用于将第二步的label数据生成相应的格式
import os

import shutil
from tqdm import tqdm


step=2
data='train'
#第一步模型
if step ==1:
    pass
# 第二步模型
elif step==2:
    if data == 'train':
        # dirpath = r'/home/igi/media/yhy/casecade/train_data/step2_labels'  # 原来存放txt文件的目录
        dirpath = r'/home/igi/media/czf/LJJJ_Fittings/val_step2/labels'
        # newdir = r'../xiushi_true/labels/train'  # 修改label后形成的txt目录
        newdir = r'/home/igi/media/czf/LJJJ_Fittings/val_step2/labels_yolo'  # 修改label后形成的txt目录
        photodir = r'/home/igi/media/czf/LJJJ_Fittings/val_step2/images'
        # photonewdir = r'../xiushi/images/train/'
    elif data =='val':
        pass
else:
    pass
if not os.path.exists(newdir):
    os.makedirs(newdir)


def xml_txt_step2():

    # 逐个拿出文件
    for fp in os.listdir(dirpath):
        lines=open(os.path.join(dirpath,fp)).readlines()
        if not len(lines):
            os.remove(os.path.join(dirpath,fp))
            os.remove(os.path.join(photodir,fp.replace('txt','jpg')))
            continue
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

# 将第一行拿出是图片尺寸

# 将后续行拿出是目标类别和位置尺寸


if __name__ == '__main__':
    xml_txt_step2()