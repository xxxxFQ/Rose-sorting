import os
path = r'D:\python\mmlab\mmdeploy-0.8.0-windows-amd64-cuda11.1-tensorrt8.2.3.0\sdk\lib'#'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\lib\x64'
files = os.listdir(path)
for i in files:
    if i[-4:] == '.lib':
        print(i)