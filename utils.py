import cv2
import torch
import numpy as np
import os.path as osp
import PIL.Image as Image


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


def cv2_save_rgb(path, img):
    H, W, C = img.shape
    if C == 3:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    elif C == 1:
        cv2.imwrite(path, img)
    

def cv2_load_rgb(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def cv2_resize_img_aspect(img, max_size=960, pad_to_64=True):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        if h > w:
            new_h = max_size
            new_w = int(w * (max_size / h))
        else:
            new_w = max_size
            new_h = int(h * (max_size / w))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
    if pad_to_64:
        h, w = img.shape[:2]
        new_h = ((h + 63) // 64) * 64
        new_w = ((w + 63) // 64) * 64
        pad_h = new_h - h
        pad_w = new_w - w

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_REFLECT)
        
    return img


def cv2_resize_img(img, new_w, new_h):
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img
    
    
def cv2_erode_image(mask, beta=2):
    C = mask.shape[-1]
    kernel = np.ones((beta, beta), np.uint8) # 创建腐蚀操作的核，大小为 (beta, beta)
    if C == 3:
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        eroded_gray_mask = cv2.erode(gray_mask, kernel, iterations=1)
        eroded_mask = cv2.cvtColor(eroded_gray_mask, cv2.COLOR_GRAY2BGR)
    else:
        eroded_mask = cv2.erode(mask, kernel, iterations=1) # 进行腐蚀操作
    return eroded_mask


def mask_to_binary(mask, rgb_dim=True):
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_mask, 200, 1, cv2.THRESH_BINARY)
    binary_array = binary_image[...,np.newaxis].astype(np.uint8)
    if rgb_dim:
        binary_array = binary_array.repeat(3, axis=2)
        
    return binary_array
        

def blend_ic_light(mask, resized_fg, ic_results, threshold):
    blended_ic_results = []
    for i, ic_res in enumerate(ic_results):
        diff = ((ic_res - resized_fg) * mask).astype(np.uint8)
        min_diff, max_diff = np.min(diff, axis=(0,1)), np.max(diff, axis=(0,1))
        weight_ic_fg = (diff - min_diff) / (max_diff - min_diff + 1e-6)
        weight_ic_fg[weight_ic_fg > threshold] = threshold
        weight_fg = 1 - weight_ic_fg
        
        # 计算混合后的图像
        blended = (ic_res * weight_ic_fg + resized_fg * weight_fg).astype(np.uint8)
        ic_res[mask==1] = blended[mask==1]
        blended_ic_results.append(ic_res)
        # cv2_save_rgb(f'results/ic_{i}.jpg', oringin_ic.astype(np.uint8))
        # cv2_save_rgb(f'results/blend_{i}.jpg', ic_fg.astype(np.uint8))
    return blended_ic_results


def blend_ic_light_bg(mask, resized_fg, resized_bg, ic_results, threshold):
    blended_ic_results = []
    for i, ic_res in enumerate(ic_results):
        diff = ((ic_res - resized_fg) * mask).astype(np.uint8)
        min_diff, max_diff = np.min(diff, axis=(0,1)), np.max(diff, axis=(0,1))
        weight_ic_fg = (diff - min_diff) / (max_diff - min_diff + 1e-6)
        weight_ic_fg[weight_ic_fg > threshold] = threshold
        weight_fg = 1 - weight_ic_fg
        
        diff = ((ic_res - resized_bg) * (1 - mask)).astype(np.uint8)
        min_diff, max_diff = np.min(diff, axis=(0,1)), np.max(diff, axis=(0,1))
        weight_ic_bg = (diff - min_diff) / (max_diff - min_diff + 1e-6)
        weight_ic_bg[weight_ic_bg > threshold] = threshold
        weight_bg = 1 - weight_ic_bg
        
        # 计算混合后的图像
        blended_fg = (ic_res * weight_ic_fg + resized_fg * weight_fg).astype(np.uint8)
        blended_bg = (ic_res * weight_ic_bg + resized_bg * weight_bg).astype(np.uint8)
        ic_res = blended_fg * mask + blended_bg * (1 - mask)
        blended_ic_results.append(ic_res.astype(np.uint8))

    return blended_ic_results



if __name__ == '__main__':
    mask = cv2_load_rgb(osp.join('results','aittor0_mask.jpg')) 
    image = cv2_load_rgb(osp.join('results','aittor0.jpg')) 
    bin_mask = mask_to_binary(mask)
    result = 127 + (image.astype(np.float32) - 127 + 0.0) * bin_mask
    result = result.clip(0, 255).astype(np.uint8) # result背景部分变成灰色
    cv2_save_rgb(osp.join('results','aittor0_fg.jpg'), result)
    