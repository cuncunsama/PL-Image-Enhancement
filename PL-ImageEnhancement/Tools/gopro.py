import os
import shutil

def copy_blur_images(src_root, dest_root):
    if not os.path.exists(dest_root):
        os.makedirs(dest_root)

    for root, dirs, files in os.walk(src_root):
        if os.path.basename(root) == 'blur':
            parent_folder = os.path.basename(os.path.dirname(root))
            for file in files:
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(dest_root, f"{parent_folder}_{file}")
                shutil.copy2(src_file_path, dest_file_path)
                print(f"Copied {src_file_path} to {dest_file_path}")

src_root = 'C:\\Users\\yxq\\datasets\\GoPro_Large\\test'  # 替换为你的GoPro数据集的路径
dest_root = 'C:\\Users\\yxq\\Desktop\\GoPro\\val\\blur'  # 替换为你想要保存图像的文件夹路径

copy_blur_images(src_root, dest_root)
