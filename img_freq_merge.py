#!/usr/bin/env python3

import numpy as np
import skimage.transform

from utils import read_img, write_img

in_filename_1 = "test-1.png"
in_filename_2 = "test-1_simplify_out0.png"
in_filename_3 = "image (4).png"
out_filename = "./out.png"

strength = 1


def main():
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


if __name__ == "__main__":
    main()