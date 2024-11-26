import os
import cv2
import numpy as np
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from joblib import Parallel, delayed
from PIL import Image
from yxq.data.utils import create_lmdb_from_folder


def generate_patches(inp_dir, flag, suffix, size, overlap, n_core):
    '''
    flag: cv2.IMREAD_COLOR(1), cv2.IMREAD_GRAYSCALE(0)
    '''
    files = natsorted(glob(os.path.join(inp_dir, suffix)))

    out_dir = inp_dir + str(size) 
    os.makedirs(out_dir, exist_ok=True)

    def crop_file(file_):
        img = cv2.imread(file_, flag)
        h, w = img.shape[0], img.shape[1]
        h_pad = max(0, size-h)
        w_pad = max(0, size-w)
        img = cv2.copyMakeBorder(img, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
        h, w = img.shape[0], img.shape[1]
        filename = os.path.splitext(os.path.split(file_)[-1])[0]
        num_patch = 0

        h1 = list(np.arange(0, h - size + 1, size - overlap))
        w1 = list(np.arange(0, w - size + 1, size - overlap))

        h1.append(h-size) if h1[-1] < h-size else None
        w1.append(w-size) if w1[-1] < w-size else None
        
        for i in h1:
            for j in w1:
                num_patch += 1
                if img.ndim == 3:
                    patch = img[i:i+size, j:j+size,:]
                else:
                    patch = img[i:i+size, j:j+size]

                savename = os.path.join(out_dir, filename + '-' + str(num_patch) + '.png')
                cv2.imwrite(savename, patch)
        
    Parallel(n_jobs=n_core)(delayed(crop_file)(file_) for file_ in tqdm(files))

def add_gaussian_noise(folder, sigma):
    files = natsorted(glob(os.path.join(folder, '*.*')))
    out_dir = folder + "_sigma"+ str(sigma)
    os.makedirs(out_dir, exist_ok=True)

    def process_file(file_):
        img = cv2.imread(file_)
        noise = np.random.normal(0, sigma, img.shape)
        noisy_img = img + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        filename = os.path.splitext(os.path.split(file_)[-1])[0]
        savename = os.path.join(out_dir, filename + '_noisy.png')
        cv2.imwrite(savename, noisy_img)

    Parallel(n_jobs=-1)(delayed(process_file)(file_) for file_ in tqdm(files))

if __name__ == "__main__":
    # add_gaussian_noise('C:\\Users\\yxq\\project\\PL\\datasets\\train\\DUKE\\lq256', 50)
    # generate_patches('C:\\Users\\yxq\\project\\PL\\datasets\\gopro_deblur\\train\\blur', 1, '*.png', 256, 0, 10)
    # create_lmdb_from_folder('C:\\Users\\yxq\\project\\PL\\datasets\\gopro_deblur\\train\\blur256')
    generate_patches('C:\\Users\\yxq\\datasets\\GoPro\\train\\blur', -1, '*.png', 256, 0, 10)
    generate_patches('C:\\Users\\yxq\\datasets\\GoPro\\train\\sharp', -1, '*.png', 256, 0, 10)
    # create_lmdb_from_folder('C:\\Users\\yxq\\datasets\\GoPro\\train\\blur')
    # create_lmdb_from_folder('C:\\Users\\yxq\\datasets\\GoPro\\train\\sharp')
    create_lmdb_from_folder('C:\\Users\\yxq\\datasets\\GoPro\\train\\blur256')
    create_lmdb_from_folder('C:\\Users\\yxq\\datasets\\GoPro\\train\\sharp256')
    pass
