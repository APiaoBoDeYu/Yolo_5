
import pandas as pd
#读取第一列、第二列、第四列
# df = pd.read_excel(r'/home/igi/media/yhy/ljm/yolov5-develop/class_num.csv',sheet_name='TestUserLogin')
# data = df.values
# print(data)

label_num=open(r'/home/igi/media/yhy/ljm/yolov5-develop/class_num.txt')
label_dir={}
for line in label_num.readlines():
    curLine = line.split(" :")
    label_dir[curLine[0]]=curLine[1]

print(label_dir)

# with open(r'/home/igi/media/yhy/ljm/yolov5-develop/class_num.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     label_num=open(r'/home/igi/media/yhy/ljm/yolov5-develop/class_num.txt')
#
#     rows=[row for row in reader]
#
#
#     print(rows)


