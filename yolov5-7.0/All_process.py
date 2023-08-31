import os
import shutil
import sys

print('开始划分数据！')
res = os.system('python check_file.py')
shutil.rmtree('runs/train-cls/color')
if res != 0:
    print('划分失败！')
    sys.exit(-1)
print('划分结束，开始训练！')
res = os.system('python classify/train.py --exist-ok --epochs 300 --name color')
if res != 0:
    print('训练失败！')
    sys.exit(-1)
print('训练结束，开始转换wts模型！')
res = os.system('python gen_wts.py -w runs/train-cls/color/weights/best.pt -o runs/train-cls/color/weights/color-best.wts')
if res != 0:
    print('转换wts模型失败！')
    sys.exit(-1)
print('转换wts模型结束，开始转换TensorRT模型！')
res = os.system('D:/VSProject/tensorrtx-yolov5-v7.02/tensorrtx-yolov5-v7.0/yolov5/build/Release/yolov5_cls.exe -s D:/VSProject/yolov5-7.0/runs/train-cls/color/weights/color-best.wts D:/VSProject/yolov5-7.0/runs/train-cls/color/weights/color_best.engine s')
if res != 0:
    print('转换Tensorrt失败！')
    sys.exit(-1)
print('转换结束，可以拷贝文件！')