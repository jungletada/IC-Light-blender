import gc
import os
import cv2
import torch
import shutil
from glob import glob
from math import ceil, floor

import numpy as np
import os.path as osp
import PIL.Image as Image
import skimage
from numba import njit


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1) # (b, c, h, w) -> (b, h, w, c) 
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
    h = h.movedim(-1, 1) # (b, h, w, c) -> (b, c, h, w)
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


def cv2_resize_img_aspect(img, max_size=1024, pad_to_64=True):
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


def cv2_resize_and_crop_bg(image, new_h, new_w):
    h, w = image.shape[:2]
    original_aspect = w / h
    target_aspect = new_w / new_h

    if original_aspect > target_aspect:
        new_width = int(h * target_aspect)
        offset = (w - new_width) // 2
        cropped_image = image[:, offset:offset + new_width]
    else:
        new_height = int(w / target_aspect)
        offset = (h - new_height) // 2
        cropped_image = image[offset:offset + new_height, :]
    
    resized_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image


def cv2_resize_img(img, new_h, new_w):
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img
    
    
def cv2_erode_image(mask, kernel_size=(3, 3)):
    C = mask.shape[-1]
    kernel = np.ones(kernel_size, np.uint8) # 创建腐蚀操作的核，大小为 (beta, beta)
    if C == 3:
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        eroded_gray_mask = cv2.erode(gray_mask, kernel, iterations=1)
        eroded_mask = cv2.cvtColor(eroded_gray_mask, cv2.COLOR_GRAY2BGR)
    else:
        eroded_mask = cv2.erode(mask, kernel, iterations=1) # 进行腐蚀操作
    return eroded_mask


def cv2_morphologyEx(mask, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return opened_mask


def mask_to_binary(mask, rgb_dim=True):
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_mask, 127, 1, cv2.THRESH_BINARY)
    binary_array = binary_image[...,np.newaxis].astype(np.uint8)
    if rgb_dim:
        binary_array = binary_array.repeat(3, axis=2)
    return binary_array
        

def blend_ic_light(resized_mask, resized_fg, ic_results, blend_value=None):
    blended_ic_results = []
    if blend_value is not None:
        for i, ic_res in enumerate(ic_results):
            diff = ((ic_res - resized_fg) * resized_mask).astype(np.uint8)
            min_diff, max_diff = np.min(diff, axis=(0,1)), np.max(diff, axis=(0,1))
            weight_ic_fg = (diff - min_diff) / (max_diff - min_diff + 1e-6)
            weight_ic_fg[weight_ic_fg > blend_value] = blend_value
            weight_fg = 1 - weight_ic_fg
            
            blended = (ic_res * weight_ic_fg + resized_fg * weight_fg).astype(np.uint8)
            ic_res[resized_mask==1] = blended[resized_mask==1]
            blended_ic_results.append(ic_res)
    else:
        return ic_results
        
    return blended_ic_results


def blend_ic_light_bg(resized_fg, resized_bg, resized_mask, ic_results, blend_value_fg=None, blend_value_bg=None):
    blended_ic_results = []
    
    if blend_value_fg is None and blend_value_bg is None:
        return ic_results
    
    for i, ic_res in enumerate(ic_results):    
        if blend_value_fg is not None:
            diff = ((ic_res - resized_fg) * resized_mask).astype(np.uint8)
            min_diff, max_diff = np.min(diff, axis=(0,1)), np.max(diff, axis=(0,1))
            weight_ic_fg = (diff - min_diff) / (max_diff - min_diff + 1e-6)
            weight_ic_fg[weight_ic_fg > blend_value_fg] = blend_value_fg
            weight_fg = 1 - weight_ic_fg
            blended_fg = (ic_res * weight_ic_fg + resized_fg * weight_fg).astype(np.uint8) 
        else:
            blended_fg = ic_res
        
        if blend_value_bg is not None:
            diff = ((ic_res - resized_bg) * (1 - resized_mask)).astype(np.uint8)
            min_diff, max_diff = np.min(diff, axis=(0,1)), np.max(diff, axis=(0,1))
            weight_ic_bg = (diff - min_diff) / (max_diff - min_diff + 1e-6)
            weight_ic_bg[weight_ic_bg > blend_value_bg] = blend_value_bg
            weight_bg = 1 - weight_ic_bg
            blended_bg = (ic_res * weight_ic_bg + resized_bg * weight_bg).astype(np.uint8)
        else: 
            blended_bg = ic_res
        
        ic_res = blended_fg * resized_mask + blended_bg * (1 - resized_mask)
        blended_ic_results.append(ic_res.astype(np.uint8))

    return blended_ic_results


def floor_even(x):
    if isinstance(x, tuple):
        return tuple(floor_even(y) for y in x)
    if isinstance(x, list):
        return [floor_even(y) for y in x]
    return x // 2 * 2


# Does not copy img and alpha
def trim_img(img, alpha, eps, *, pad=0):
    original_shape = alpha.shape

    trim_t = 0
    while np.all(alpha[trim_t, :] <= eps):
        trim_t += 1
    trim_b = alpha.shape[0] - 1
    while np.all(alpha[trim_b, :] <= eps):
        trim_b -= 1
    trim_l = 0
    while np.all(alpha[:, trim_l] <= eps):
        trim_l += 1
    trim_r = alpha.shape[1] - 1
    while np.all(alpha[:, trim_r] <= eps):
        trim_r -= 1

    trim_b += 1
    trim_r += 1

    trim_t = max(trim_t - pad, 0)
    trim_b = min(trim_b + pad, alpha.shape[0])
    trim_l = max(trim_l - pad, 0)
    trim_r = min(trim_r + pad, alpha.shape[1])

    trims = (trim_t, trim_b, trim_l, trim_r)
    return original_shape, trims


def untrim_img(img, alpha, original_shape, trims):
    trim_t, trim_b, trim_l, trim_r = trims
    new_img = np.zeros((original_shape[0], original_shape[1], 3))
    new_alpha = np.zeros(original_shape)
    new_img[trim_t:trim_b, trim_l:trim_r, :] = img
    new_alpha[trim_t:trim_b, trim_l:trim_r] = alpha
    return new_img, new_alpha


def get_tiles(img, tile_inner_size, pad_size, *, wrap_x=False, wrap_y=False):
    tile_outer_size = tile_inner_size + pad_size * 2

    max_row = ceil(img.shape[0] / tile_inner_size)
    max_col = ceil(img.shape[1] / tile_inner_size)
    img_padded_h = max_row * tile_inner_size
    img_padded_w = max_col * tile_inner_size
    pad_t = floor((img_padded_h - img.shape[0]) / 2)
    pad_b = img_padded_h - img.shape[0] - pad_t
    pad_l = floor((img_padded_w - img.shape[1]) / 2)
    pad_r = img_padded_w - img.shape[1] - pad_l
    img_full = np.pad(
        img,
        [(pad_t + pad_size, pad_b + pad_size), (0, 0), (0, 0)],
        "wrap" if wrap_y else "reflect",
    )
    img_full = np.pad(
        img_full,
        [(0, 0), (pad_l + pad_size, pad_r + pad_size), (0, 0)],
        "wrap" if wrap_x else "reflect",
    )

    tiles = []
    for i in range(max_row):
        for j in range(max_col):
            idx_t = i * tile_inner_size
            idx_b = idx_t + tile_outer_size
            idx_l = j * tile_inner_size
            idx_r = idx_l + tile_outer_size
            tiles.append(img_full[idx_t:idx_b, idx_l:idx_r, :])
    tiles = np.stack(tiles)

    max_row_col = (max_row, max_col)
    pads = (pad_t, pad_b, pad_l, pad_r)
    return tiles, max_row_col, pads


def get_batch(tiles, batch_size):
    idx = 0
    while idx < tiles.shape[0]:
        batch = tiles[idx : idx + batch_size]
        batch = batch.transpose(0, 3, 1, 2)
        idx += batch.shape[0]
        print(f"Tile {idx}/{tiles.shape[0]}")
        yield batch


def merge_img(tiles, tile_inner_size, pad_size, max_row_col, pads, scale_shift=(1, 0)):
    max_row, max_col = max_row_col
    pad_t, pad_b, pad_l, pad_r = pads
    scale, shift = scale_shift

    tile_outer_size = tile_inner_size + pad_size * 2
    scaled_inner_size = tile_inner_size * scale
    scaled_outer_size = tile_outer_size * scale

    img = np.empty((max_row * scaled_outer_size, max_col * scaled_outer_size, 3))
    for idx, tile in enumerate(tiles):
        i = idx // max_col
        j = idx % max_col

        idx_t = i * scaled_inner_size
        idx_b = idx_t + scaled_inner_size
        idx_l = j * scaled_inner_size
        idx_r = idx_l + scaled_inner_size
        tile_l = pad_size * scale - shift
        tile_r = tile_l + scaled_inner_size
        img[idx_t:idx_b, idx_l:idx_r, :] = tile[tile_l:tile_r, tile_l:tile_r, :]

    img = img[
        pad_t * scale : (max_row * tile_inner_size - pad_b) * scale,
        pad_l * scale : (max_col * tile_inner_size - pad_r) * scale,
        :,
    ]
    return img


# Inplace
def randomize(img, n_bins):
    delta = 1 / n_bins
    img += delta * (np.random.rand(*img.shape) - 0.5)


# Inplace
@njit
def quantize(img, n_bins):
    H, W, C = img.shape
    for i in range(H):
        for j in range(W):
            for k in range(C):
                x0 = img[i, j, k]
                x = round(x0 * n_bins) / n_bins
                x = min(max(x, 0), 1)
                r = x0 - x
                img[i, j, k] = x

                # Do not dither alpha
                if k == 3:
                    continue

                if i == H - 1:
                    if j < W - 1:
                        img[i, j + 1, k] += r
                else:
                    if j == 0:
                        img[i, j + 1, k] += r / 2
                        img[i + 1, j, k] += r / 2
                    elif j == W - 1:
                        img[i + 1, j - 1, k] += r / 2
                        img[i + 1, j, k] += r / 2
                    else:
                        img[i, j + 1, k] += r / 2
                        img[i + 1, j - 1, k] += r / 4
                        img[i + 1, j, k] += r / 4


# Inplace
@njit
def quantize_adapt(img):
    H, W, C = img.shape
    for i in range(H):
        for j in range(W):
            for k in range(C):
                x0 = img[i, j, k]
                if x0 > 0.5:
                    n_bins = 15
                elif x0 > 0.25:
                    n_bins = 31
                elif x0 > 0.125:
                    n_bins = 63
                elif x0 > 0.0625:
                    n_bins = 127
                elif x0 > 0.03125:
                    n_bins = 255

                x = round(x0 * n_bins) / n_bins
                x = min(max(x, 0), 1)
                r = x0 - x
                img[i, j, k] = x

                # Do not dither alpha
                if k == 3:
                    continue

                if i == H - 1:
                    if j < W - 1:
                        img[i, j + 1, k] += r
                else:
                    if j == 0:
                        img[i, j + 1, k] += r / 2
                        img[i + 1, j, k] += r / 2
                    elif j == W - 1:
                        img[i + 1, j - 1, k] += r / 2
                        img[i + 1, j, k] += r / 2
                    else:
                        img[i, j + 1, k] += r / 2
                        img[i + 1, j - 1, k] += r / 4
                        img[i + 1, j, k] += r / 4

    
def read_img(
    filename,
    *,
    swap_rb=False,
    gamma=1,
    signed=True,
    scale=None,
    noise=0,
    return_alpha=False,
):
    # Use cv2 to support 16 bit image
    img = np.fromfile(filename, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    img = skimage.img_as_float32(img)

    alpha = None

    if img.ndim == 3:
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
        else:
            assert img.shape[2] == 3
        # Remove alpha channel
        img = img[:, :, :3]
    else:
        assert img.ndim == 2
        # Convert grayscale to RGB
        img = np.repeat(img[:, :, None], 3, axis=2)

    if swap_rb:
        assert img.ndim == 3
        # BGR -> RGB
        img = img[:, :, ::-1]

    img **= gamma

    if signed:
        # [0, 1] -> [-1, 1]
        img = img * 2 - 1

    if scale is not None:
        img *= scale
        if alpha is not None:
            alpha *= scale

    if noise:
        rng = np.random.default_rng()
        img += rng.normal(scale=noise, size=img.shape)

    if return_alpha:
        return img, alpha
    else:
        return img


def write_img(
    filename,
    img,
    *,
    alpha=None,
    swap_rb=False,
    signed=True,
    scale=None,
    output_gray=False,
    output_8_bit=True,
    quant_bit=0,
):
    if scale is not None:
        img /= scale
        if alpha is not None:
            alpha /= scale

    if signed:
        # [-1, 1] -> [0, 1]
        img = (img + 1) / 2
        if alpha is not None:
            alpha = (alpha + 1) / 2

    if swap_rb:
        assert img.ndim == 3
        # RGB -> BGR
        img = img[:, :, ::-1]

    if output_gray and img.ndim == 3:
        img = img.mean(axis=2, keepdims=True)

    if img.ndim == 2:
        img = img[:, :, None]

    if alpha is not None:
        if alpha.ndim == 2:
            alpha = alpha[:, :, None]
        img = np.concatenate([img, alpha], axis=2)

    print("Quantizing...")
    if output_8_bit and quant_bit == 0:
        quant_bit = 8
    if quant_bit == "adapt":
        img = 1 - img
        quantize_adapt(img)
        img = 1 - img
    elif quant_bit > 0:
        n_bins = 2**quant_bit - 1
        randomize(img, n_bins)
        quantize(img, n_bins)
    else:
        img = np.clip(img, 0, 1)
    if output_8_bit:
        img = skimage.img_as_ubyte(img)
    else:
        img = skimage.img_as_uint(img)

    print("Encoding...")
    ret, img = cv2.imencode(os.path.splitext(filename)[1], img)
    assert ret is True

    print("Writing...")
    img.tofile(filename)


def do_imgs(
    fun,
    model_filenames,
    in_patterns,
    *,
    out_suffix=None,
    out_extname=None,
    tmp_filename=None,
):
    if isinstance(model_filenames, str):
        model_filenames = [model_filenames]
    elif model_filenames is None:
        model_filenames = [None]

    if isinstance(in_patterns, str):
        in_patterns = [in_patterns]

    in_filenames = []
    for in_pattern in in_patterns:
        in_filename = glob(in_pattern)
        if not in_filename:
            print(f"Warning: File not found: {in_pattern}")
        in_filenames += in_filename
    if not in_filenames:
        print("Warning: No input file")

    for model_filename in model_filenames:
        if model_filename:
            import onnxruntime as rt

            print(model_filename)
            sess = rt.InferenceSession(
                model_filename,
                providers=[
                    # "TensorrtExecutionProvider",
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
            )

            if out_suffix is None:
                _out_suffix = (
                    "_" + os.path.splitext(os.path.basename(model_filename))[0]
                )
            else:
                _out_suffix = out_suffix
        else:
            sess = None
            assert out_suffix is not None
            _out_suffix = out_suffix

        for in_filename in in_filenames:
            print(in_filename)

            basename, extname = os.path.splitext(in_filename)
            if isinstance(_out_suffix, tuple):
                out_filename = basename.replace(_out_suffix[0], _out_suffix[1])
                if len(_out_suffix) >= 3:
                    out_filename += _out_suffix[2]
            else:
                out_filename = basename + _out_suffix
            if out_extname is None:
                out_extname = extname
            out_filename += out_extname

            if tmp_filename:
                shutil.copy2(in_filename, tmp_filename)
                fun(sess, tmp_filename, tmp_filename)
                shutil.move(tmp_filename, out_filename)
            else:
                fun(sess, in_filename, out_filename)

        if sess is not None:
            del sess
            gc.collect()
    

def merge_image(in_filename_1, in_filename_2, in_filename_3, out_filename, strength=1):
    img_1 = read_img(in_filename_1, swap_rb=True, signed=False)
    img_2 = read_img(in_filename_2, swap_rb=True, signed=False)
    img_3 = read_img(in_filename_3, swap_rb=True, signed=False)

    shape = img_1.shape[:2]
    img_2 = skimage.transform.resize(img_2, shape)
    img_3 = skimage.transform.resize(img_3, shape)

    img_1 -= strength * img_2
    img_1 += strength * img_3
    img_1 = np.clip(img_1, 0, 1)

    write_img(out_filename, img_1, swap_rb=True, signed=False)