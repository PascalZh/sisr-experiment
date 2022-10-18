import si_algo_model
import cv2
import os
from sr_utils import PSNR, SSIM

def dataloader():
    downscale = si_algo_model.Downscale()
    for file in os.listdir("images"):
        print(f"Loading ./images/{file}...")
        HR = cv2.imread(f"images/{file}").astype(float)
        LR = downscale(HR)
        for lpatch, hpatch in si_algo_model.LR_HR_patch_pairs(LR, HR):
            yield lpatch, hpatch

# GT = cv2.imread("images/LenaRGB.bmp")
# LR = downscale(GT)
# HR = si.predict(LR).astype('uint8')

# def assess_quality(GT, HR):
#     print(f"GT: shape={GT.shape}, dtype={GT.dtype}, HR: shape={HR.shape}, dtype={HR.dtype}")
#     print(f"PSNR = {PSNR(GT, HR)}, SSIM = {SSIM(GT, HR, full=True, channel_axis=2)[0]}")

# assess_quality(GT, HR)

# cv2.imwrite("images/LenaRGB2x.bmp", HR)

