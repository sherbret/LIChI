#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Name : LIChI
# Copyright (C) Inria,  Sébastien Herbreteau, Charles Kervrann, All Rights Reserved, 2022, v1.0.

import torch
from torchvision.io import read_image, write_png
from lichi import LIChI
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--sigma", type=float, dest="sigma", help="Standard deviation of the noise (noise level). Should be between 0 and 50.", default=15)
parser.add_argument("--in", type=str, dest="path_in", help="Path to the image to denoise (PNG or JPEG).", default="./test_images/cameraman.png")
parser.add_argument("--out", type=str, dest="path_out", help="Path to save the denoised image.", default="./denoised.png")
parser.add_argument("--add_noise", action='store_true', help="Add artificial Gaussian noise to the image.")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Reading
img = read_image(args.path_in)[None, :, :, :].float().to(device)
img_noisy = img + args.sigma * torch.randn_like(img) if args.add_noise else img

# Denoising
model = LIChI()
t = time.time()
if args.sigma <= 10:
	den = model(img_noisy, sigma=args.sigma, constraints='affine', method='n2n', p1=9, p2=6, k1=16, k2=64, w=65, s=3, M=6)
elif args.sigma <= 30:
	den = model(img_noisy, sigma=args.sigma, constraints='affine', method='n2n', p1=11, p2=6, k1=16, k2=64, w=65, s=3, M=9)
else:
	den = model(img_noisy, sigma=args.sigma, constraints='affine', method='n2n', p1=13, p2=6, k1=16, k2=64, w=65, s=3, M=11)
print("Time elapsed:", round(time.time() - t, 3), "seconds")
den = den.clip(0, 255)

# Performance in PSNR
if args.add_noise:
    psnr = 10*torch.log10(255**2 / torch.mean((den - img)**2))
    print("PSNR:", round(float(psnr), 2), "dB")

# Writing
write_png(den[0, :, :, :].byte(), args.path_out)
