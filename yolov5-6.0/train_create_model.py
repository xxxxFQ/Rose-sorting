import os
import sys

res = os.system('python train.py --epochs 300 --name open --exist-ok')
if res != 0:
    print('训练失败！')
    sys.exit(-1)
print('训练成功，开始转换wts格式！')
res = os.system('python gen_wts.py -w runs/train/open/weights/best.pt -o runs/train/open/weights/open_best.wts')
if res != 0:
    print('转换失败！')
    sys.exit(-1)
print('开始转换Tensorrt！')
res = os.system('D:/VSProject/tRTyolov5s6/yolov5/Debug/yolov5.exe -s D:/VSProject/yolov5-6.0/runs/train/open/weights/open_best.wts D:/VSProject/yolov5-6.0/runs/train/open/weights/openRose.engine s')
if res != 0:
    print('转换失败!')
    sys.exit(-1)
print('转换成功！')