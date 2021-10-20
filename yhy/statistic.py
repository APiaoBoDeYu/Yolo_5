#*-*coding=utf-8*-*
import os,cv2,shutil
import xml.dom.minidom
from tqdm import tqdm
path = "/home/igi/media/czf/LJJJ_Fittings/val/xml/"#'../xml'#'./train_data/Annotations'
# path = "/home/igi/media/yhy/jinjuxiushi/test_dataset/xml/"#'../xml'#'./train_data/Annotations'





className=[]
className_num = []
fzc_num = 0
attr_name_list =[]#['LX','ZT']
attrlist=[0,0,0,0]


xml_list=[]
pbar=tqdm(range(10000))
single_xml_list=[]
lx_unknown_list=[]
num_lx=0
for roots,dir,files in os.walk(path):
    for file in files:
        #print(file)
        if file.split('.')[-1] == 'xml':
            #print(file)
            xml_list.append(file)
            respond_pic = os.path.join(roots.replace('xml','photo'),file.split('.')[0]+'.JPG')
            if not os.path.isfile(respond_pic):
                single_xml_list.append(file)
            try:
                pbar.update(1)
                xml_path_abs=os.path.join(roots,file)
                dom = xml.dom.minidom.parse(xml_path_abs)
                root = dom.documentElement
            except:
                print(roots + '/' + file)
                continue
            labellist = root.getElementsByTagName('object')
            #img = cv2.imread(respond_pic)
            containFlag = False
            for label in labellist:
                strname = str(label.getElementsByTagName('name')[0].firstChild.data)
                # if not strname == 'JYZ_C':
                #     continue
                for attr_name in attr_name_list:
                   if len(label.getElementsByTagName(attr_name)) >0:
                       attr_value =str(label.getElementsByTagName(attr_name)[0].firstChild.data)
                       #print(attr_value)
                       # if attr_value.__contains__('unknown') or attr_value.__contains__('no'):
                       #     lx_unknown_list.append(file)
                       #     continue
                       strname += '_'+ attr_value
                if strname.__contains__('GTYW'):
                    containFlag = True
                # if strname in badlist:
                #     print(file)
                if not strname in className:
                    className.append(strname)
                    className_num.append(1)
                else:
                    for i in range(0, len(className)):
                        if className[i] == strname:
                            className_num[i] = className_num[i] + 1
            if False:#containFlag:
                shutil.copy(os.path.join(roots, file), os.path.join(tmppath, file))
                shutil.copy(os.path.join(roots, file.replace('xml','JPG' )).replace('xml','photo'),
                            os.path.join(tmppath, file.replace('xml','JPG')))
                # print(xml_path_abs)
                # img = cv2.imread(respond_pic)
                # # if str_name.__contains__("lj") or str_name.__contains__("LJ"):
                # xmin = int(label.getElementsByTagName('bndbox')[0].getElementsByTagName('xmin')[0].firstChild.data)
                # ymin = int(label.getElementsByTagName('bndbox')[0].getElementsByTagName('ymin')[0].firstChild.data)
                # xmax = int(label.getElementsByTagName('bndbox')[0].getElementsByTagName('xmax')[0].firstChild.data)
                # ymax = int(label.getElementsByTagName('bndbox')[0].getElementsByTagName('ymax')[0].firstChild.data)
                # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                # #cv2.rectangle(img,(),(),(255),2)
                # cv2.putText(img, str(strname), (xmin, ymin - 40), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
                # cv2.imwrite(tmppath+file.split('.')[0]+'.JPG',img)


# print("It contains {} xmls!".format(len(xml_list)))
#
# print("It contains {} single xmls!".format(len(single_xml_list)))
# print("It contains {} unknown xmls!".format(len(lx_unknown_list)))
print('\n')
for i in range(len(className)):
    print("{} :{}".format(className[i],className_num[i]))
