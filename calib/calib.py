import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from PIL import ImageEnhance
from tqdm import tqdm, trange
import os, sys
import rawpy
import zivid
import yaml

width, height = 11, 8
square_size = 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# convert 16bit to 8bit
def convert_16bit_to_8bit(img):
    img = img.astype(np.float32)
    img = img / 257
    img = img.astype(np.uint8)
    return img

# rgb to srgb
def rgb_to_srgb(image, max_value = -1):
    if max_value == -1:
        if image.dtype == np.uint8:
            max_value = 255
        elif image.dtype == np.uint16:
            max_value = 65535
        elif image.dtype == np.float32:
            max_value = 1
        else:
            raise ValueError("Unknown image type.")
    ret = image.astype(np.float32)
    ret /= max_value
    ret = np.where(ret > 0.0031308, 1.055 *
                   np.power(ret.clip(min=0.0031308), 1 / 2.4) - 0.055, 12.92 * ret)
    ret *= max_value
    ret = ret.astype(image.dtype)
    ret = np.maximum(ret, 0)
    ret = np.minimum(ret, max_value)
    return ret


def generate_masked_image(image, p1, p2):
    # generate a mask
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask = cv2.rectangle(mask, p1, p2, (255, 255, 255), -1)
    # masked_image = cv2.bitwise_and(image, mask)
    mask_preview = cv2.rectangle(np.zeros(image.shape, dtype=np.uint8), p1, p2, (255, 0, 0), -1)
    mask_preview = cv2.addWeighted(image, 0.8, mask_preview, 0.2, 0)
    mask_preview = cv2.cvtColor(mask_preview, cv2.COLOR_RGB2BGR)
    return mask_preview

# get roi from image with mouse drag
def get_rect(im, title='get_rect'): # (a,b) = get_rect(im, title='get_rect')
    mouse_params = {'tl': None, 'br': None, 'current_pos': None, 'released_once': False}
    ratio = 1000 / im.shape[0]
    im = cv2.resize(im, (0, 0), fx=ratio, fy=ratio)
    cv2.namedWindow(title)
    cv2.moveWindow(title, 1500, 200)

    def onMouse(event, x, y, flags, param):
        param['current_pos'] = (x, y)
        if param['tl'] is not None and not (flags & cv2.EVENT_FLAG_LBUTTON):
            param['released_once'] = True
        if flags & cv2.EVENT_FLAG_LBUTTON:
            if param['tl'] is None:
                param['tl'] = param['current_pos']
            elif param['released_once']:
                param['br'] = param['current_pos']

    cv2.setMouseCallback(title, onMouse, mouse_params)
    cv2.imshow(title, im)

    while mouse_params['br'] is None:
        im_draw = np.copy(im)
        if mouse_params['tl'] is not None:
            cv2.rectangle(im_draw, mouse_params['tl'],
                mouse_params['current_pos'], (255, 0, 0))
        cv2.imshow(title, im_draw)
        _ = cv2.waitKey(10)
    cv2.destroyWindow(title)

    tl = (min(mouse_params['tl'][0], mouse_params['br'][0]),
        min(mouse_params['tl'][1], mouse_params['br'][1]))
    br = (max(mouse_params['tl'][0], mouse_params['br'][0]),
        max(mouse_params['tl'][1], mouse_params['br'][1]))

    tl = (int(tl[0] / ratio), int(tl[1] / ratio))
    br = (int(br[0] / ratio), int(br[1] / ratio))

    return (tl, br)

# load zivid zdf file
def load_zdf_rgb(filename):
    with zivid.Application() as app:
        frame = zivid.Frame(filename)
        # get rgb image
        rgb = frame.point_cloud().copy_data("rgba")[:,:,:3]
        rgb = rgb[::-1, ::-1, :]
        rgb = rgb[:, 160:-160, :]
        print(f"Load zivid img with shape {rgb.shape}")
    return rgb

# postprocess zivid image and find chessboard corners
def find_chessboard_corners_zivid(rgb, tlbr=None, preview_path="../data/ref", ratio=8733/1200):
    if tlbr is None:
        tl, br = get_rect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), title='get_rect')
    else:
        tl, br = tlbr

    x1, y1, x2, y2 = tl[0], tl[1], br[0], br[1]
    h, w = y2-y1, x2-x1
    print(f"ROI: ({x1}, {y1}), ({x2}, {y2})")
    roi_preview = rgb.copy()
    cv2.rectangle(roi_preview, (x1, y1), (x2, y2), (255, 0, 0), 5)

    h0, h1, h2, h3 = 0, 0.34, 0.67, 1
    w0, w1, w2, w3 = 0.06, 0.36, 0.64, 0.90
    w4, w5, w6 = 0.24, 0.5, 0.75
    w7, w8, w9, w10 = 0.04, 0.33, 0.64, 0.96

    cam1_p1 = [
        [[y1+h0*h, x1+w0*w], [y1+h1*h, x1+w1*w]],
        [[y1+h0*h, x1+w1*w], [y1+h1*h, x1+w2*w]],
        [[y1+h0*h, x1+w2*w], [y1+h1*h, x1+w3*w]],

        [[y1+h1*h, x1], [y1+h2*h, x1+w4*w]],
        [[y1+h1*h, x1+w4*w], [y1+h2*h, x1+w5*w]],
        [[y1+h1*h, x1+w5*w], [y1+h2*h, x1+w6*w]],
        [[y1+h1*h, x1+w6*w], [y1+h2*h, x1+w]],

        [[y1+h2*h, x1+w7*w], [y1+h3*h, x1+w8*w]],
        [[y1+h2*h, x1+w8*w], [y1+h3*h, x1+w9*w]],
        [[y1+h2*h, x1+w9*w], [y1+h3*h, x1+w10*w]],
    ]

    cam1_p1 = np.asarray(cam1_p1, dtype=np.int32)
    cam1_p1 = cam1_p1[:, :, ::-1]
    for b in cam1_p1:
        cv2.rectangle(roi_preview, b[0], b[1], (0, 0, 255), 3)
    cv2.imwrite(os.path.join(preview_path, "roi_preview_zivid.png"), roi_preview)

    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width*square_size:square_size, 0:height*square_size:square_size].T.reshape(-1, 2)

    objpoints1 = [] # 3d point in real world space
    imgpoints1 = [] # 2d points in image plane

    preview = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    preview = cv2.resize(preview, (0, 0), fx=ratio, fy=ratio)
    for i in trange(10):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        p1, p2 = cam1_p1[i]
        sub_gray = gray[p1[1]:p2[1], p1[0]:p2[0]]
        ret, corners = cv2.findChessboardCorners(sub_gray, (width, height), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            objpoints1.append(objp)
            corners2 = cv2.cornerSubPix(sub_gray, corners, (11, 11), (-1, -1), criteria)
            corners2 += np.array(p1)
            corners2 = corners2 * ratio
            imgpoints1.append(corners2)
            cv2.drawChessboardCorners(preview, (width, height), corners2, ret)
        else:
            print("not found")

    cv2.imwrite(os.path.join(preview_path, "cb", f"single1.jpg"), preview)

    return objpoints1, imgpoints1
    

# load fuji RAF file
def load_fuji_rgb(filename):
    with rawpy.imread(filename) as raw:
        rgb = raw.postprocess(gamma=(1, 1), use_camera_wb=True, no_auto_bright=True, output_bps=16)[:8733, :11644]
    print(f"Load fuji img with shape {rgb.shape}")
    return rgb

# postprocess fuji image and find chessboard corners
def find_chessboard_corners_fuji(rgb, tlbr=None, preview_path="../data/ref"):
    if tlbr is None:
        tl, br = get_rect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), title='get_rect')
    else:
        tl, br = tlbr

    x1, y1, x2, y2 = tl[0], tl[1], br[0], br[1]
    h, w = y2-y1, x2-x1
    print(f"ROI: ({x1}, {y1}), ({x2}, {y2})")
    roi_preview = rgb.copy()
    cv2.rectangle(roi_preview, (x1, y1), (x2, y2), (255, 0, 0), 5)

    h0, h1, h2, h3 = 0, 0.33, 0.67, 1
    w0, w1, w2, w3 = 0.05, 0.38, 0.66, 0.95
    w4, w5, w6 = 0.25, 0.5, 0.75
    w7, w8, w9, w10 = 0.05, 0.36, 0.65, 0.93

    cam2_p1 = [
        [[y1+h0*h, x1+w0*w], [y1+h1*h, x1+w1*w]],
        [[y1+h0*h, x1+w1*w], [y1+h1*h, x1+w2*w]],
        [[y1+h0*h, x1+w2*w], [y1+h1*h, x1+w3*w]],

        [[y1+h1*h, x1], [y1+h2*h, x1+w4*w]],
        [[y1+h1*h, x1+w4*w], [y1+h2*h, x1+w5*w]],
        [[y1+h1*h, x1+w5*w], [y1+h2*h, x1+w6*w]],
        [[y1+h1*h, x1+w6*w], [y1+h2*h, x1+w]],

        [[y1+h2*h, x1+w7*w], [y1+h3*h, x1+w8*w]],
        [[y1+h2*h, x1+w8*w], [y1+h3*h, x1+w9*w]],
        [[y1+h2*h, x1+w9*w], [y1+h3*h, x1+w10*w]],
    ]

    cam2_p1 = np.asarray(cam2_p1, dtype=np.int32)
    cam2_p1 = cam2_p1[:, :, ::-1]

    for b in cam2_p1:
        cv2.rectangle(roi_preview, b[0], b[1], (0, 0, 255), 3)
    cv2.imwrite(os.path.join(preview_path, "roi_preview_fuji.png"), roi_preview)

    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width*square_size:square_size, 0:height*square_size:square_size].T.reshape(-1, 2)

    objpoints2 = [] # 3d point in real world space
    imgpoints2 = [] # 2d points in image plane

    preview = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    for i in trange(10):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        p1, p2 = cam2_p1[i]
        sub_gray = gray[p1[1]:p2[1], p1[0]:p2[0]]
        ret, corners = cv2.findChessboardCorners(sub_gray, (width, height), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            objpoints2.append(objp)
            corners2 = cv2.cornerSubPix(sub_gray, corners, (11, 11), (-1, -1), criteria)
            corners2 += np.array(p1)
            imgpoints2.append(corners2)
            cv2.drawChessboardCorners(preview, (width, height), corners2, ret)
        else:
            print("not found")

    cv2.imwrite(os.path.join(preview_path, "cb", f"single2.jpg"), preview)

    return objpoints2, imgpoints2

# Calibrate camera
def calibrate_camera(objpoints1, imgpoints1, objpoints2, imgpoints2, path="../data"):
    ret1, mtx1, dist1, _, _ = cv2.calibrateCamera(objpoints1, imgpoints1, (11644, 8733), None, None)
    print(f"ret1: {ret1}")
    print(f"mtx1: {mtx1}")
    print(f"dist1: {dist1}")

    ret2, mtx2, dist2, _, _ = cv2.calibrateCamera(objpoints2, imgpoints2, (11644, 8733), None, None)
    print(f"ret2: {ret2}")
    print(f"mtx2: {mtx2}")
    print(f"dist2: {dist2}")

    if len(imgpoints1) != 10 or len(imgpoints2) != 10:
        raise ValueError("not enough images")

    retval, k1, d1, k2, d2, R, T, E, F = cv2.stereoCalibrate(objpoints1, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2, (11644, 8733), flags=cv2.CALIB_FIX_INTRINSIC+cv2.CALIB_FIX_K3+cv2.CALIB_FIX_K4+cv2.CALIB_FIX_K5)

    print("retval", retval)
    print("R", R)
    print("T", T)

    data = {
        "k1": k1.tolist(),
        "d1": d1.tolist(),
        "k2": k2.tolist(),
        "d2": d2.tolist(),
        "R": R.tolist(),
        "T": T.tolist(),
        "E": E.tolist(),
        "F": F.tolist(),
    }

    with open(os.path.join(path, "calib_result.yaml"), "w") as f:
        yaml.dump(data, f)