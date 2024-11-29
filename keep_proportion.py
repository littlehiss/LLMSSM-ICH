import os
import cv2
import math
import numpy as np

def get_dir(src_point, rot_rad):
    sn = math.sin(rot_rad)
    cs = math.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return [src_result[0], src_result[1]]


def get_3rd_point(a, b):
    direct = [a[0] - b[0], a[1] - b[1]]
    direct = [b[0] - direct[1], b[1] + direct[0]]
    return direct

def get_affine_transform(center, scale, rot, output_size, inv=False):
    scale_tmp = scale[0]
    src_w = scale_tmp
    dst_w = output_size[0]
    dst_h = output_size[1]

    half_1 = 0.5
    shift = [0, 0]
    rot_rad = 3.1415926 * rot / 180
    src_point = [0, src_w * -half_1]

    src_dir = get_dir(src_point, rot_rad)

    dst_dir = [0, dst_w * -half_1]

    src = []
    dst = []

    src.append([center[0] + scale_tmp * shift[0], center[1] + scale_tmp * shift[1]])
    src.append([center[0] + src_dir[0] + scale_tmp * shift[0], center[1] + src_dir[1] + scale_tmp * shift[1]])

    dst.append([dst_w * half_1, dst_h * half_1])
    dst.append([dst_w * half_1 + dst_dir[0], dst_h * half_1 + dst_dir[1]])

    src.append(get_3rd_point(src[0], src[1]))
    dst.append(get_3rd_point(dst[0], dst[1]))

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def get_affine_trans_image(img, network_input_size):
    height, width = img.shape[:2]
    center = [width / 2.0, height / 2.0]
    s = max(height, width) * 1.0
    scale = [s, s]
    rolate = 0
    trans = get_affine_transform(center, scale, rolate, network_input_size, False)
    transed_dstImage = cv2.warpAffine(img, trans, (network_input_size[1], network_input_size[0]))
    center_scale = {"input": transed_dstImage, "center": center, "scale": scale, "r": 0.0}
    return center_scale
