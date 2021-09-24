# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 09:26:55 2021
@author: Elena Corbetta, Daniele Ancora

Functions to build multi-view measurements.
"""

import numpy as np
import scipy.ndimage
import functions.pyphret_functions as pf

#%%
'''Generation of a 3D matrix containing different rotations of the input 2D PSF,
one for each view, according to the angles defined by angle_rot (in degrees)'''
def rotated_psf(psf, angle_rot):

    psf_rot = np.zeros((angle_rot.shape[0], psf.shape[0], psf.shape[1]))
    
    for i in np.arange(angle_rot.shape[0]):
        if angle_rot[i] == 0:
            psf_rot[i,:,:] = psf
        else:
            psf_rot[i,:,:] = scipy.ndimage.rotate(psf, angle_rot[i], reshape=False)
    return psf_rot



'''Generation of a simulated measurement by applying blurring and Poisson noise 
to the input image '''
def blurred_noisy_image(image, psf, lambd):
    
    noise = np.random.poisson(lam=lambd, size=image.shape)   
    image_blur = np.abs(pf.my_convolution(image, psf))
    image_blur = (2**16) * image_blur/image_blur.max()
    image_blur = image_blur + noise - lambd
    image_blur[image_blur<0] = 0
    
    return image_blur


''' Generation of a multiview measurement by applying different PSFs 
to the input image and adding Poisson noise to each view'''
def views_blurred_noisy_image(image, psf_rot, lambd):
    image_views = np.zeros((psf_rot.shape[0], image.shape[0], image.shape[1]))
    for i in np.arange(psf_rot.shape[0]):
        image_views[i,:,:] = blurred_noisy_image(image, psf_rot[i], lambd)
        
    return image_views



'''Independent autocorrelation of each slice of a 3D matrix'''
def autocorrelateViews(stackViews):
    autocorr = np.zeros_like(stackViews)
    for i in range(stackViews.shape[0]):
        autocorr[i,:,:] = pf.my_autocorrelation(stackViews[i,:,:])
    return autocorr
        
