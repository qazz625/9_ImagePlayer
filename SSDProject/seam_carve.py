import numpy as np
from PIL import Image
import seam_carving

# def calc_energy(img):
#     filter_du = np.array([
#         [1.0, 2.0, 1.0],
#         [0.0, 0.0, 0.0],
#         [-1.0, -2.0, -1.0],
#     ])
#     filter_du = np.stack([filter_du] * 3, axis=2)

#     filter_dv = np.array([
#         [1.0, 0.0, -1.0],
#         [2.0, 0.0, -2.0],
#         [1.0, 0.0, -1.0],
#     ])
#     filter_dv = np.stack([filter_dv] * 3, axis=2)

#     img = img.astype('float32')
#     convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))
#     energy_map = convolved.sum(axis=2)

#     return energy_map

# def crop_c(img, scale_c):
#     r, c, _ = img.shape
#     new_c = int(scale_c * c)

#     for i in range(c - new_c):
#         img = carve_column(img)

#     return img

# def crop_r(img, scale_r):
#     img = np.rot90(img, 1, (0, 1))
#     img = crop_c(img, scale_r)
#     img = np.rot90(img, 3, (0, 1))
#     return img

# def carve_column(img):
#     r, c, _ = img.shape

#     M, backtrack = minimum_seam(img)
#     mask = np.ones((r, c), dtype=bool)

#     j = np.argmin(M[-1])
#     for i in reversed(range(r)):
#         mask[i, j] = False
#         j = backtrack[i, j]

#     mask = np.stack([mask] * 3, axis=2)
#     img = img[mask].reshape((r, c - 1, 3))
#     return img

# def minimum_seam(img):
#     r, c, _ = img.shape
#     energy_map = calc_energy(img)

#     M = energy_map.copy()
#     backtrack = np.zeros_like(M, dtype=np.int32)

#     for i in range(1, r):
#         for j in range(0, c):
#             if j == 0:
#                 idx = np.argmin(M[i-1, j:j + 2])
#                 backtrack[i, j] = idx + j
#                 min_energy = M[i-1, idx + j]
#             else:
#                 idx = np.argmin(M[i - 1, j - 1:j + 2])
#                 backtrack[i, j] = idx + j - 1
#                 min_energy = M[i - 1, idx + j - 1]

#             M[i, j] += min_energy

#     return M, backtrack

# def seam_carving(in_filename):
#     scale=0.5
#     img = imread(in_filename)
#     out=crop_r(img,scale)
#     out=crop_c(out,scale)
#     imwrite(in_filename, out)

def seam_carv_crop(img):
    src = img
    src_h, src_w, _ = src.shape
    dst = seam_carving.resize(
        src,  # source image (rgb or gray)
        size=(src_w-0.2*src_w, src_h-0.2*src_h),  # target size
        energy_mode="forward",  # choose from {backward, forward}
        order="width-first",  # choose from {width-first, height-first}
        keep_mask=None,  # object mask to protect from removal
    )
    return dst

def seam_carv_expand(img):
    src = img
    src_h, src_w, _ = src.shape
    dst = seam_carving.resize(
        src,  # source image (rgb or gray)
        size=(src_w+0.2*src_w, src_h+0.2*src_h),  # target size
        energy_mode="forward",  # choose from {backward, forward}
        order="width-first",  # choose from {width-first, height-first}
        keep_mask=None,  # object mask to protect from removal
    )
    return dst