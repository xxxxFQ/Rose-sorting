import os
import random
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def check_img_txt(yolov5_result_root):
    '''
    查看img有没有对应的label文件
    '''
    miss_file = []
    img_file = os.listdir(yolov5_result_root)
    txt_file = os.listdir(f'{yolov5_result_root}/labels')

    for i in img_file:
        if i.replace('jpg', 'txt') not in txt_file:
            # print(i)
            miss_file.append(i)

    return miss_file, len(txt_file)


def check_label(yolov5_result_labels_root, cls):
    '''
    查看label是否正确
    '''
    wrong_file = []

    txt_file_list = os.listdir(yolov5_result_labels_root)
    for txt_file in txt_file_list:
        txt = open(f'{yolov5_result_labels_root}/{txt_file}')
        data = txt.readline()
        if int(data[0]) != cls:
            wrong_file.append(txt_file)

    return wrong_file, len(txt_file_list)


def cut_img(root, txt, img_root, save_root):
    # yolo标注数据文件名为786_rgb_0616.txt
    # with open(f'{root}/labels/{txt}', 'r') as f:
    with open(f'{root}/{txt}', 'r') as f:
        temp = f.read()
        temp = temp.split()
    # ['1', '0.43906', '0.52083', '0.34687', '0.15']

    img = cv2.imread(f'{img_root}/{txt}'.replace('txt', 'jpg'))
    h, w, _ = img.shape

    # 根据第1部分公式进行转换
    x_, y_, w_, h_ = eval(temp[1]), eval(temp[2]), eval(temp[3]), eval(temp[4])

    x1 = int(w * x_ - 0.5 * w * w_)
    x2 = int(w * x_ + 0.5 * w * w_)
    y1 = int(h * y_ - 0.5 * h * h_)
    y2 = int(h * y_ + 0.5 * h * h_)

    img_cut = img[y1:y2, x1:x2]
    cv2.imwrite(f'{save_root}/{txt}'.replace('txt', 'jpg'), img_cut)


def change_label(root, file, id, save_root):
    txt = open(f'{root}/{file}')
    data = txt.readlines()
    # new_data = []
    for i in data:
        if int(i[0]) != id:
            # print(file)

            new_data = f'{id}' + i[1:]
            new_txt = open(f'{save_root}/{file}', 'a')
            new_txt.write(new_data)
        else:
            new_txt = open(f'{save_root}/{file}', 'a')
            new_data = i
            new_txt.write(new_data)


def split_array(array):
    # 计算划分的比例
    total_length = len(array)
    train_length = int(total_length * 0.7)
    # val_length = int(total_length * 0.2)

    # 随机选择元素进行划分
    train_array = random.sample(array, train_length)
    val_array = [item for item in array if item not in train_array]
    # val_array = random.sample(remaining_array, val_length)
    # test_array = [item for item in remaining_array if item not in val_array]

    return train_array, val_array#, test_array


def vis_img_color_histogram(img_root, save_root):
    # # 读取图像
    # image_path = 'yolooutput/exp/cutimg/Img_0_1.jpg'  # 图像文件路径
    # image = Image.open(image_path)
    #
    # # 转换为RGB模式
    # image = image.convert('RGB')
    #
    # # 获取颜色直方图
    # histogram = image.histogram()
    #
    # # 显示直方图
    # plt.figure()
    # plt.title('Color Histogram')
    # plt.xlabel('Color Value')
    # plt.ylabel('Frequency')
    # plt.plot(histogram)
    # plt.show()

    # 读取图像
    image_path = img_root
    image = Image.open(image_path)

    # 将图像转换为RGB模式（如果不是RGB图像）
    image = image.convert('RGB')

    # 将图像转换为NumPy数组
    image_array = np.array(image)
    file_name = os.path.basename(img_root)
    # 分离每个颜色通道
    red_channel = image_array[:, :, 0]
    green_channel = image_array[:, :, 1]
    blue_channel = image_array[:, :, 2]

    # 计算每个通道的直方图
    red_histogram = np.histogram(red_channel.flatten(), bins=256, range=(0, 256))
    green_histogram = np.histogram(green_channel.flatten(), bins=256, range=(0, 256))
    blue_histogram = np.histogram(blue_channel.flatten(), bins=256, range=(0, 256))

    # 绘制红色通道直方图
    plt.bar(range(len(red_histogram[0])), red_histogram[0], color='red', alpha=0.7)
    plt.ylim(0, 10000)
    plt.xlabel('Color Value')
    plt.ylabel('Frequency')
    plt.title('Red Channel Histogram')
    plt.savefig(f'{save_root}/red-{file_name}')
    # plt.show()
    plt.clf()

    # 绘制绿色通道直方图
    plt.bar(range(len(green_histogram[0])), green_histogram[0], color='green', alpha=0.7)
    plt.ylim(0, 10000)
    plt.xlabel('Color Value')
    plt.ylabel('Frequency')
    plt.title('Green Channel Histogram')
    plt.savefig(f'{save_root}/green-{file_name}')
    # plt.show()
    plt.clf()

    # 绘制蓝色通道直方图
    plt.bar(range(len(blue_histogram[0])), blue_histogram[0], color='blue', alpha=0.7)
    plt.ylim(0, 10000)
    plt.xlabel('Color Value')
    plt.ylabel('Frequency')
    plt.title('Blue Channel Histogram')
    plt.savefig(f'{save_root}/blue-{file_name}')
    # plt.show()
    plt.clf()
    plt.close()


def vis_img_hsv_histogram(img_root, save_root):
    # 读取图像
    image_path = img_root
    image = cv2.imread(image_path)

    # 将图像转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    file_name = os.path.basename(image_path)
    # 分离H、S、V通道
    h, s, v = cv2.split(hsv_image)

    # 计算每个通道的直方图
    h_hist = cv2.calcHist([h[np.where(v >= 50)]], [0], None, [180], [0, 180])  # [np.where(v >= 40)]
    s_hist = cv2.calcHist([s[np.where(v >= 50)]], [0], None, [256], [0, 256])
    v_hist = cv2.calcHist([v[np.where(v >= 50)]], [0], None, [256], [0, 256])

    # 绘制H通道的直方图
    plt.figure()
    plt.plot(h_hist, color='r')
    plt.ylim(0, 100000)
    plt.title('H Channel Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.savefig(f'{save_root}/H-{file_name}')
    plt.clf()

    # 绘制S通道的直方图
    plt.plot(s_hist, color='g')
    plt.ylim(0, 10000)
    plt.title('S Channel Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.savefig(f'{save_root}/S-{file_name}')
    plt.clf()

    # 绘制V通道的直方图
    plt.plot(v_hist, color='b')
    plt.ylim(0, 10000)
    plt.title('V Channel Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.savefig(f'{save_root}/V-{file_name}')
    plt.clf()
    plt.close()
    # 显示直方图
    # plt.show()


# def show_v_channel():
#     # 读取图像
#     image_path = 'yolooutput/exp/cutimg/pink/Img_0_4.jpg'
#     image = cv2.imread(image_path)
#
#     # 将图像转换为HSV颜色空间
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
#     # 分离V通道
#     v = hsv_image[:, :, 2]
#
#     # 显示V通道的图像
#     plt.imshow(v, cmap='gray')
#     plt.title('V Channel')
#     plt.axis('off')
#
#     # 显示图像
#     plt.show()

def img_rgb_hsv_nor(img_path):
    ''' 图片的rgb通道和hsv通道归一化后求和'''
    # 读取图像
    image_path = img_path
    image = cv2.imread(image_path)
    # print(image.shape)
    # 将图像转换为RGB颜色空间
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 将图像转换为HSV颜色空间
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 归一化RGB通道
    normalized_rgb = image_rgb / 255.0

    # 归一化HSV通道
    normalized_hsv = np.copy(image_hsv)
    normalized_hsv = normalized_hsv.astype(np.float64)
    normalized_hsv[:, :, 0] = (normalized_hsv[:, :, 0] / 179.0)  # 归一化H通道，取值范围为[0, 1]
    normalized_hsv[:, :, 1:] = (normalized_hsv[:, :, 1:] / 255.0)  # 归一化S和V通道，取值范围为[0, 1]

    # 对各自通道求和
    sum_rgb = np.sum(normalized_rgb, axis=(0, 1))
    sum_hsv = np.sum(normalized_hsv, axis=(0, 1))
    all_pixel = image.shape[0] * image.shape[1]
    # print("RGB Channel Sum:", sum_rgb)
    # print("HSV Channel Sum:", sum_hsv)
    return sum_rgb / all_pixel, sum_hsv / all_pixel

def  cal_pink_rate(file):
    # 加载输入图像
    img = cv2.imread(file)

    # 将图像转换为HSV颜色空间
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # plt.imshow(hsv_img)
    # plt.show()
    # 定义粉色的上下边界（在HSV颜色空间）
    lower_pink = np.array([150.0, 53.75, 158.5])
    upper_pink = np.array([175.0, 255.0, 255.0])

    # 创建一个掩码，只保留在粉色范围内的像素
    mask = cv2.inRange(hsv_img, lower_pink, upper_pink)

    # 计算掩码中非零元素的数量（即粉色像素的数量）
    pink_pixels = cv2.countNonZero(mask)

    # 计算粉色像素在整个图像中所占的百分比
    total_pixels = img.shape[0] * img.shape[1]
    pink_percent = (pink_pixels / total_pixels) * 100

    # 打印输出结果
    print(f"粉色像素占比: {pink_percent}%")

    # # 显示输出掩码
    # cv2.imshow("Mask", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    root = 'data/open'
    class_open_dir = os.listdir(root)
    save_root = 'data/opening_degree'
    if os.path.exists(save_root) is True:
        shutil.rmtree(save_root)
    os.mkdir(save_root)
    for class_open in class_open_dir:
        ''' 划分数据集 '''
        image_root = f'{root}/{class_open}'
        img_train_save_root = f'{save_root}/images/train/{class_open}'
        img_val_save_root = f'{save_root}/images/val/{class_open}'
        labels_train_save_root = f'{save_root}/labels/train/{class_open}'
        labels_val_save_root = f'{save_root}/labels/val/{class_open}'
        if os.path.exists(img_train_save_root) is False:
            os.makedirs(img_train_save_root)
            os.makedirs(img_val_save_root)
            os.makedirs(labels_train_save_root)
            os.makedirs(labels_val_save_root)
        txt_file_list = os.listdir(f'{image_root}/label')
        if os.path.exists(f'{image_root}/labels') is False:
            os.makedirs(f'{image_root}/labels')
        for txt_name in txt_file_list:
            if class_open == '2' or class_open == '2度':
                change_label(f'{image_root}/label', txt_name, 0, f'{image_root}/labels')
            else:
                change_label(f'{image_root}/label', txt_name, 1, f'{image_root}/labels')
        filelist = os.listdir(image_root)
        train, val = split_array(filelist)
        for i in filelist:
            if i.endswith('.jpg'):
                txt = i.replace("jpg", "txt")
                txt_list = os.listdir(f'{image_root}/labels')
                if txt in txt_list:
                    if i in train:
                        shutil.copy(f'{image_root}/{i}', f'{img_train_save_root}/{i}')
                        shutil.copy(f'{image_root}/labels/{txt}', f'{labels_train_save_root}/{txt}')
                    elif i in val:
                        shutil.copy(f'{image_root}/{i}', f'{img_val_save_root}/{i}')
                        shutil.copy(f'{image_root}/labels/{txt}', f'{labels_val_save_root}/{val}')

