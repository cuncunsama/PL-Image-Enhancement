import scipy.io
import numpy as np
from PIL import Image
import os

# 定义读取.mat文件并保存图像的函数
def mat_to_image(mat_file, output_dir):
    key = 'BenchmarkNoisyBlocksSrgb'
    inputs = scipy.io.loadmat(mat_file)
    inputs = inputs[key]
    print(f'inputs.shape = {inputs.shape}')
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            in_block = inputs[i, j, :, :, :]
            print(f'in_block.shape = {in_block.shape}')

            if np.max(in_block) <= 1:
                print("in_block is normalized")
                in_block = (in_block * 255).astype(np.uint8)
            else:
                print("in_block is not normalized")
                in_block = in_block.astype(np.uint8)
            img = Image.fromarray(in_block)
            output_path = os.path.join(output_dir, f'image_{i}_{j}.png')
            img.save(output_path)

# 示例使用
mat_file = r'C:\Users\yxq\project\sidd_kaggle_submit-main\BenchmarkNoisyBlocksSrgb.mat'
output_dir = r'C:\Users\yxq\datasets\SIDD\test'
mat_to_image(mat_file, output_dir)
