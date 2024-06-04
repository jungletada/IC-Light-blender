import cv2
import numpy as np
import os.path as osp
import PIL.Image as Image


def cv2_save_rgb(path, img):
    H, W, C = img.shape
    if C == 3:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    elif C == 1:
        cv2.imwrite(path, img)
    

def cv2_load_rgb(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def blend_ic_light(mask, resized_fg, ic_fg_results, threshold):
    blended_ic_results = []
    for i, ic_fg in enumerate(ic_fg_results):

        diff = (ic_fg - resized_fg) * mask.astype(np.uint8)
        min_diff, max_diff = np.min(diff, axis=(0,1)), np.max(diff, axis=(0,1))
        weight_ic_fg = (diff - min_diff) / (max_diff - min_diff + 1e-6)
        weight_ic_fg[weight_ic_fg > threshold] = threshold
        weight_resized_fg = 1 - weight_ic_fg
        
        # 计算混合后的图像
        blended = (ic_fg * weight_ic_fg + resized_fg * weight_resized_fg).astype(np.uint8)
        ic_fg[mask==1] = blended[mask==1]
        blended_ic_results.append(ic_fg)
        cv2_save_rgb(f'results/blend_{i}.jpg', ic_fg.astype(np.uint8))
    return blended_ic_results
        

if __name__ == '__main__':
    mask = cv2_load_rgb(osp.join('results','mask.png')) 
    mask = mask / 255.
    mask = mask.astype(np.uint8)
    resized_fg = cv2_load_rgb(osp.join('results','fg_image.jpg'))
    num_samples = 4
    ic_fg_results = []
    for i in range(num_samples):
        ic_fg = cv2_load_rgb(osp.join('results',f'iclight_image_{i}.jpg'))
        ic_fg_results.append(ic_fg)
    blend_ic_light(mask, resized_fg, ic_fg_results)