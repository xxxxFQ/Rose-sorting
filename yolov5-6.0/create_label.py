import os
import shutil
import sys

print('开始生成label！')
file_root = 'data/open'
for floder_name in os.listdir(file_root):
    file = f'{file_root}/{floder_name}'
    res = os.system(f'python detect.py --source {file} --name {floder_name} --exist-ok --nosave --save-txt ')
    if res != 0:
        print('出现错误！')
        sys.exit(-1)
    label_root = f'runs/detect/{floder_name}/labels'
    save_root = f'{file}/label'
    if os.path.exists(save_root) is False:
        os.mkdir(save_root)
    for labelfile_name in os.listdir(label_root):
        label_file = f'{label_root}/{labelfile_name}'
        shutil.move(label_file, f'{save_root}/{labelfile_name}')
print('生成label结束，开始划分数据集！')
res = os.system('python check_file.py')
if res != 0:
    print('划分失败！')
    sys.exit(-1)
print('划分成功，请检查数据集！')

