import cv2
import numpy as np
import pandas as pd
import scipy as sp
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
from skimage.color import rgb2gray


#== Parameters =======================================================================
DATA_PATH = 'data-raw/'
TARGET_SIZE = 224
OUT_PATH = 'data-'+str(TARGET_SIZE)+'/'
FILETYPES = ['jpg', 'png']

BLUR = 3
CANNY_THRESH_1 = 30
CANNY_THRESH_2 = 200
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (1.0,1.0,1.0) # In BGR format

#== Processing =======================================================================
def remove_background(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # Find contours in edges, sort by area
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    # Create empty mask, draw filled polygon on it corresponding to largest contour
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    # Smooth mask, then blur it
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

    # Blend masked img into MASK_COLOR background
    mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices,
    img         = img.astype('float32') / 255.0                 #  for easy blending

    masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
    masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit

    return cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)



img_paths = [f for f in listdir(DATA_PATH) if (isfile(join(DATA_PATH, f)) and f[-3:] in FILETYPES)]

count = 1;
for img_path in img_paths:
    # Load image & remove background
    full_path = DATA_PATH + img_path
    img = cv2.imread(full_path)
    img_nobg = remove_background(img) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Find image bounding box
    img_coords = np.argwhere(rgb2gray(img_nobg) < 1)
    bb = [img_coords.min(axis=0), img_coords.max(axis=0)]

    # Trim whitespace & resample to target size
    img = img[bb[0][0]:bb[1][0],bb[0][1]:bb[1][1]]

    final_img = np.ones((TARGET_SIZE, TARGET_SIZE, 3))
    img_size = img.shape;
    if img_size[0] < img_size[1]:
        img_ratio = img_size[0] / img_size[1]
        img_resized = resize(img, (int(TARGET_SIZE*img_ratio), TARGET_SIZE), mode='constant')

        h = img_resized.shape[0]
        start = int((TARGET_SIZE - h) / 2)
        final_img[start:start+h,0:,0:] = img_resized
    else:
        img_ratio = img_size[1] / img_size[0]
        img_resized = resize(img, (TARGET_SIZE, int(TARGET_SIZE*img_ratio)), mode='constant')

        w = img_resized.shape[1]
        start = int((TARGET_SIZE - w) / 2)
        final_img[0:,start:start+w,0:] = img_resized

    mpimg.imsave(OUT_PATH + 'frog-' + str(count) + '.png', final_img)
    count += 1;
