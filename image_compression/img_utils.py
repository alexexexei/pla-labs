from PIL import Image
import numpy as np
import os
import glob

img_extensions = ('png', 'jpg', 'jpeg', 'webp', 'bmp')

def get_image_paths(folder_path: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, folder_path)

    image_paths = []
    for ext in img_extensions:
        image_paths.extend(glob.glob(os.path.join(full_path, '*.' + ext)))

    return image_paths

def read_image(path: str):
    return Image.open(path)

def save_img(img: Image, path: str):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(path)

def calc_compr_val(dim: int, compr_perc: int):
    return dim - int(dim * (compr_perc / 100))

def get_compressed_svd(img: Image, compr_perc: int):
    img_arr = np.asarray(img)
    return get_compressed_svd(img_arr, compr_perc)

def get_compressed_svd(arr: list, compr_perc: int):
    U, S, V = np.linalg.svd(arr)

    compr_val = calc_compr_val(dim=S.shape[0], compr_perc=compr_perc)

    return U[:, :compr_val], S[:compr_val], V[:compr_val, :]

def compress_grayscale_img(img: Image, compr_perc: int):
    grayscale_img = img.convert('L')
    U_compr, S_compr, V_compr = get_compressed_svd(grayscale_img, compr_perc)

    return Image.fromarray(np.matrix(U_compr) * np.diag(S_compr) * np.matrix(V_compr))

def compress_rgb_img(img: Image, compr_perc: int):
    red_img = np.array(img)[:, :, 0]
    green_img = np.array(img)[:, :, 1]
    blue_img = np.array(img)[:, :, 2]

    U_r_compr, S_r_compr, V_r_compr = get_compressed_svd(red_img, compr_perc)
    U_g_compr, S_g_compr, V_g_compr = get_compressed_svd(green_img, compr_perc)
    U_b_compr, S_b_compr, V_b_compr = get_compressed_svd(blue_img, compr_perc)

    red_compr = np.matrix(U_r_compr) * np.diag(S_r_compr) * np.matrix(V_r_compr)
    green_compr = np.matrix(U_g_compr) * np.diag(S_g_compr) * np.matrix(V_g_compr)
    blue_compr = np.matrix(U_b_compr) * np.diag(S_b_compr) * np.matrix(V_b_compr)

    rgb_compr = np.zeros((np.array(img).shape[0], np.array(img).shape[1], 3))
    rgb_compr[:, :, 0] = red_compr
    rgb_compr[:, :, 1] = green_compr
    rgb_compr[:, :, 2] = blue_compr
    rgb_compr = np.clip(rgb_compr, 0, 255)
    rgb_compr = np.around(rgb_compr).astype(int)

    return Image.fromarray(rgb_compr.astype(np.uint8))