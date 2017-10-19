import tomosaic
import dxchange
import numpy as np
import tomopy
import time


def pad_sinogram(sino, length, mean_length=40, mode='edge'):

    length = int(length)
    res = np.zeros([sino.shape[0], sino.shape[1] + length * 2])
    res[:, length:length+sino.shape[1]] = sino
    if mode == 'edge':
        mean_left = np.mean(sino[:, :mean_length], axis=1).reshape([sino.shape[0], 1])
        mean_right = np.mean(sino[:, -mean_length:], axis=1).reshape([sino.shape[0], 1])
        res[:, :length] = mean_left
        res[:, -length:] = mean_right

    return res


def scale_images(img, min, max, dtype='uint8'):

    factor = 1
    if dtype == 'uint8':
        factor = 255
    img = (img - min) / float(max - min) * factor
    img[img < 0] = 0
    img[img > factor] = factor
    img = img.astype(dtype)
    return img

# x06 center = 248
# x07 center = pad + 705

pad_length = 1024

t0 = time.time()

for slice_st, slice_end in [(0, 100), (100, 200), (200, 300), (300, 400), (400, 512)]:
    prj, flt, drk = tomosaic.read_data_adaptive('S3_FullBrain_Mosaic_x06_y06.h5', sino=(slice_st, slice_end))
    prj = tomopy.normalize(prj, flt, drk)
    theta = tomopy.angles(prj.shape[0])
    prj = tomosaic.preprocess(prj, normalize_bg=False)
    padded = np.zeros([prj.shape[0], prj.shape[1], prj.shape[2] + pad_length * 2])
    for i in range(prj.shape[1]):
        padded[:, i, :] = pad_sinogram(np.squeeze(prj[:, i, :]), pad_length)
    prj = padded
    rec = tomopy.recon(prj, theta, center=pad_length+248, algorithm='gridrec')
    rec = scale_images(rec, 0, 0.0015)
    for i in range(rec.shape[0]):
        dxchange.write_tiff(rec[i], 'x06/recon_{:05d}'.format(slice_st+i), dtype='uint8', overwrite=True)

print time.time() - t0
t0 = time.time()

for slice_st, slice_end in [(0, 100), (100, 200), (200, 300), (300, 400), (400, 512)]:
    prj, flt, drk = tomosaic.read_data_adaptive('S3_FullBrain_Mosaic_x07_y06.h5', sino=(slice_st, slice_end))
    prj = tomopy.normalize(prj, flt, drk)
    theta = tomopy.angles(prj.shape[0])
    prj = tomosaic.preprocess(prj, normalize_bg=False)
    padded = np.zeros([prj.shape[0], prj.shape[1], prj.shape[2] + pad_length * 2])
    for i in range(prj.shape[1]):
        padded[:, i, :] = pad_sinogram(np.squeeze(prj[:, i, :]), pad_length)
    prj = padded
    rec = tomopy.recon(prj, theta, center=pad_length+705, algorithm='gridrec')
    rec = scale_images(rec, 0, 0.0015)
    for i in range(rec.shape[0]):
        dxchange.write_tiff(rec[i], 'x07/recon_{:05d}'.format(slice_st+i), dtype='uint8', overwrite=True)

print time.time() - t0


# dxchange.write_tiff(np.squeeze(prj), 'sino', dtype='float32', overwrite=True)
# tomopy.write_center(prj, theta, cen_range=(pad_length+456+245, pad_length+456+255))
# rec = tomopy.recon(prj, theta, center=pad_length+705, algorithm='gridrec')
# dxchange.write_tiff(rec, 'recon', dtype='float32', overwrite=True)