# coding=utf-8
from xml.etree.ElementTree import parse, Element
import os

classname = []
classnamenum = []
import shutil

# badlist=['JYZ_C_PWJYZ_czzs','JYZ_C_PWJYZ_czps','JYZ_C_PWJYZ_chd','JYZ_C_unknown']
badlist = []
path2 = '/home/igi/media/czf/LJJJ_Fittings/val/xml/'

XCXJ=["XCXJ1","XCXJ2","XCXJ3"]
XCXJ1=["XC","TB","ZXHZ","XGJ","CGF","SK","TX_ddx"]
XCXJ2=["CZSFL_zxhz","CZSFL_tb","CZSFL_yj","TX_sfl","CZSFL_xc"]
change_name = {"45_NZXJ_ZDX":"44_NZXJ_YYX",
               "1_Ball Eyes":"QTJJ",
               "2_Ball Clevis_Q&U":"QTJJ",
               "5_Shackles_U&UL":"ZJGB_1",
               "7_Clain Links_PH":"ZJGB_1",
               "8_Shack_ZH":"ZJGB_1",
                "10_Clevis_Z&ZS":"ZJGB_2",
                "11_Clevis_UB":"ZJGB_2",
                "12_Clevis_UBR":"ZJGB_2",
                "13_GDJJ":"ZJGB_2",
                "14_Clevis_P":"ZJGB_2",
                "15_Clevis_PD":"ZJGB_2",
                "16_Clevis_PS":"ZJGB_2",
               "21_Adjusting Plate_DB":"ZJGB_3",
                "23_Yoke Plate_L1":"Yoke_Plate",
                "24_Yoke Plate_L2":"Yoke_Plate",
                "25_Yoke Plate_L3":"Yoke_Plate",
                "26_Yoke Plate_L4":"Yoke_Plate"
               }
def modifyXMLName(folderPath, fileName):
    tree = parse(folderPath + '/' + fileName)
    root = tree.getroot()
    labels = tree.findall('./object')
    index = 0
    xcxj_num=0
    for label in labels:  # labels:

        name = label.find('name').text
        if name in change_name.keys():
            label.find('name').text = change_name[name]
        elif name=="XCXJ":
            try:
                if label.find('LX').text in XCXJ:
                    label.find('name').text = label.find('LX').text
                elif label.find('LX').text in XCXJ1:
                    label.find('name').text = "XCXJ1"
                elif label.find('LX').text in XCXJ2:
                    label.find('name').text = "XCXJ2"
                else:
                    xcxj_num+=1
                    print("找到不属于XCXJ1,2,3的XCXJ共：{}".format(xcxj_num))
            except AttributeError:
                print("这个标签没有LX属性")
            except:
                print("未知错误")
        if name in badlist:
            print(index)
            # labels.pop(index)
            root.remove(label)
        index += 1

        if not name in classname:
            classname.append(name)
            classnamenum.append(1)
        else:
            for i in range(len(classname)):
                if name == classname[i]:
                    classnamenum[i] += 1

    tree.write(path2 + '/' + fileName, encoding='utf-8')  # 此处应保存为Annotations的路径


fileList = os.listdir(path2)
for fileName in fileList:
    if not fileName.split('.')[-1] == 'xml':
        continue
    modifyXMLName(path2, fileName)
print(classname, classnamenum)
