import numpy as np
import cv2



def extract_boundary(mask, boundary_width=4):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测提取边界
    edges = cv2.Canny(mask, 100, 200)

    # 创建一个结构元素，用于扩展边界
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (boundary_width, boundary_width))

    # 使用膨胀操作扩展边界的宽度
    boundary = cv2.dilate(edges, kernel)

    return boundary



def apply_smoothing_in_region(image, mask, smoothing_function, *args, **kwargs):
    assert image.shape[:2] == mask.shape[:2], "Image and mask must have the same dimensions"

    result = image.copy()

    # 对整个图像进行平滑处理
    smoothed_image = smoothing_function(image, *args, **kwargs)

    for c in range(3):
        # 只在掩码指定的区域应用平滑效果
        result[:, :, c] = np.where(mask == 255, smoothed_image[:, :, c], image[:, :, c])

    return result

def smoothedges(image, mask):
    """
    image: ndarry: h,w,3
    mask: ndarry: h,w,3
    """
    #使用中值滤波，kernel size默认为7
    mask = extract_boundary(mask)
    return apply_smoothing_in_region(image, mask, cv2.medianBlur, 7)

image = cv2.imread('./output3.png')
mask = cv2.imread('./cookingmachine_mask.png')

res = smoothedges(image,mask)
cv2.imwrite('res.png', res)