import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

def length(px, py, qx, qy):
    return int(np.sqrt((px-qx) ** 2 + (py-qy) ** 2))

# sample image
img = np.array(Image.open("./test2022410000.png"), dtype=np.uint8)
print("Image shape : ", img.shape)

# 0
mask_0 = np.zeros(img.shape)
cv2.circle(mask_0, (203, 181), length(204,181, 204,36)-1, (255, 255, 0), -1, cv2.LINE_AA) 
mask_0 = (mask_0 == 255.).astype(np.uint8)
# 1
mask_1 = np.zeros(img.shape)
cv2.circle(mask_1, (354, 171), length(354,171, 354,26), (255, 255, 0), -1, cv2.LINE_AA) 
mask_1 = (mask_1 == 255.).astype(np.uint8)
# 2
mask_2 = np.zeros(img.shape)
cv2.circle(mask_2, (429, 187), length(429,187, 429,355)-1, (255, 255, 0), -1, cv2.LINE_AA) 
mask_2 = (mask_2 == 255.).astype(np.uint8)
# 3
mask_3 = np.zeros(img.shape)
cv2.circle(mask_3, (442, 372), length(442,372, 442,226), (255, 255, 0), -1, cv2.LINE_AA) 
mask_3 = (mask_3 == 255.).astype(np.uint8)
# 4
mask_4 = np.zeros(img.shape)
cv2.circle(mask_4, (442, 299), length(442,299, 612,298)-1, (255, 255, 0), -1, cv2.LINE_AA) 
mask_4 = (mask_4 == 255.).astype(np.uint8)
# 5
mask_5 = np.zeros(img.shape)
cv2.circle(mask_5, (292, 417), length(292,417, 296,274), (255, 255, 0), -1, cv2.LINE_AA) 
mask_5 = (mask_5 == 255.).astype(np.uint8)
# 6
mask_6 = np.zeros(img.shape)
cv2.circle(mask_6, (285, 500), length(285,500, 137,500)-2, (255, 255, 0), -1, cv2.LINE_AA) 
mask_6 = (mask_6 == 255.).astype(np.uint8)
# 7
mask_7 = np.zeros(img.shape)
cv2.circle(mask_7, (326, 493), length(326,493, 325,350), (255, 255, 0), -1, cv2.LINE_AA) 
mask_7 = (mask_7 == 255.).astype(np.uint8)

directory = os.path.join(os.getcwd(), 'data/mask')

try:
    if not os.path.exists(directory):
        os.makedirs(directory)
except OSError:
    print ('Error: Creating directory. ' +  directory)


print("Image save")
for i, mask in enumerate([mask_0, mask_1, mask_2, mask_3, mask_4, mask_5, mask_6, mask_7]):
    path = os.path.join(directory, 'mask_{}.png'.format(i))
    Image.fromarray(mask).save(path, 'png')
