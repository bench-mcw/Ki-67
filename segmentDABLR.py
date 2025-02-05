import cv2
import matplotlib.pyplot as plt

#
from skimage.color import rgb2hed
import skimage as ski

print("downloading LR into memory...")
lrecon = cv2.imread("/Volumes/Siren/Brain_data/1.PatientDirectory/221/Histology/Processed/S11_Large/SOX2/LR10_N221_S11_Large_SOX2.jpg")

print("seperating LR into HED color channels...")
hed_lrecon = rgb2hed(lrecon)
dab = hed_lrecon[:,:,2]
hem = hed_lrecon[:,:,0]

print("applying threshold to dab channel...")
window_size = 1025
thresh_lr_niblack = ski.filters.threshold_niblack(dab, window_size=window_size, k=.2)

binary_lr_niblack = dab > thresh_lr_niblack
plt.imshow(binary_lr_niblack, cmap=plt.cm.gray)