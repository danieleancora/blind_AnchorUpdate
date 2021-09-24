# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 11:18:05 2021
@author: Elena Corbetta, Daniele Ancora

Blind PSF and object reconstruction of synthetic data
"""

import numpy as np
import functions.pyphret_functions   as pf
import functions.blind_functions     as bf
import functions.multiview_functions as mf
import tifffile as tiff
import matplotlib.pyplot as plt


#%% load the dataset
image = tiff.imread('dataset/vessels01.tif')

image = image[48:239,35:226] # crop: shape=(191,191)
image = image/image.sum()

angle_rot = np.array([0,90])
lambd     = 2**8

psf     = pf.gaussian_psf(size=image.shape, alpha=[2.7,1])
psf_rot = mf.rotated_psf(psf, angle_rot)

measure = mf.views_blurred_noisy_image(image, psf_rot, lambd)
measure = measure/measure.sum()

psf_xcorr = mf.autocorrelateViews(psf_rot)
xcorr     = mf.autocorrelateViews(measure)

psf_xcorr_mean = psf_xcorr.mean(axis=0)
xcorr_mean     = xcorr.mean(axis=0)

#%% blind deconvolution
iterations_h   = 50
iterations_o   = 50
iterations_tot = 2000
step_image     = 20
kerneltype     = 'B'
alpha_prior    = [4,4]

o_prior = measure.mean(axis=0) + 0.01*measure.mean()

h_prior = mf.rotated_psf(pf.gaussian_psf(size=psf.shape, alpha=alpha_prior), angle_rot)
h_prior = h_prior.mean(axis=0)

H_prior = mf.autocorrelateViews(mf.rotated_psf(pf.gaussian_psf(size=psf.shape, alpha=alpha_prior), angle_rot))
H_prior = H_prior.mean(axis=0)


o,h,error_o,error_h = bf.blindAU(xcorr_mean, o_prior, h_prior, H_prior,
                                 iterations_o, iterations_h, iterations_tot,
                                 step_image, kerneltype)
    
# %% Plots
crop = 80


# Figure 2
plt.figure()
plt.subplot(221), plt.imshow(measure[0,:,:]), plt.title("First view")
plt.subplot(222), plt.imshow(measure[1,:,:]), plt.title("Second view")
plt.subplot(223), plt.imshow(psf_rot[0,:,:]), plt.title("PSF of the first view")
plt.subplot(224), plt.imshow(psf_rot[1,:,:]), plt.title("PSF of the second view")
plt.suptitle('Figure 2 - Synthetic sample for blind deconvolution')
plt.tight_layout()

# Figure 3
plt.figure()
plt.subplot(231), plt.imshow(image),                plt.title("Original synthetic sample")
plt.subplot(232), plt.imshow(measure.mean(axis=0)), plt.title("Synthetic measurement")
plt.subplot(233), plt.imshow(o[-1,:,:]),            plt.title("Reconstruction")
plt.subplot(234), plt.imshow(psf_rot.mean(axis=0)[crop:-crop,crop:-crop]), plt.title("Blurring kernel")
plt.subplot(235), plt.imshow(h_prior[crop:-crop,crop:-crop]),              plt.title("Kernel initial guess")
plt.subplot(236), plt.imshow(h[-1,crop:-crop,crop:-crop]),            plt.title("Reconstruction")
plt.suptitle('Figure 3 - Simulated blind AU deconvolved deautocorrelation')
plt.tight_layout()

