# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 09:43:35 2021
@author: Elena Corbetta, Daniele Ancora

Function to execute blind Anchor-Update algorithm.
It is a generalised function, working with:
    - numpy and cupy
    - 2D and 3D measurements
    - AU with kerneltype A and B
"""


######### import cupy only if installed #########
from importlib import util
cupy_enabled = util.find_spec("cupy") is not None
if cupy_enabled:
    import cupy  as cp
######### ----------------------------- #########

import numpy                  as np
import time
import functions.pyphret_functions as pf

#%%
def blindAU(signal, o_prior, h_prior, H_prior=np.float32(0),
            iterations_o=50, iterations_h=50, iterations_tot=10,
            step_image=1, kerneltype='B'):

    """

    Parameters
    ----------
    signal : ndarray, either numpy or cupy. 
        The measured auto-correlation to be inverted and deconvolved
    o_prior : ndarray, either numpy or cupy. 
        Initial guess for the object.
    h_prior : ndarray, either numpy or cupy. 
        Initial guess for the PSF in the direct space.
    H_prior : ndarray, either numpy or cupy. 
        Initial guess for the PSF in the autocorrelation space.
    iterations_o : int
        Iterations to execute at each blind cycle to deconvolve the object.
    iterations_h : int
        Iterations to execute at each blind cycle to deconvolve the PSF.
    iterations_tot : int
        Number of external blind cycles to execute.
    step_image : int
        Number of outer blind cycles to execute before saving an intermediate
        object and PSF reconstruction.
    kerneltype : string
        Type of kernel update used for the computation of AU algorithm.
        'A' --> blurring is applied in the autocorrelation space
        'B' --> blurring is applied in the direct space, then the object is 
                autocorrelated  
        The default is 'B'.
        

    Returns
    -------
    o : ndarray, either numpy or cupy.
        Deconvolved de-autocorrelated object.
    h : either numpy or cupy.
        Deconvolved de-autocorrelated PSF.
    error_o : array.
        SNR between signal and the auto-correlation of o convolved with h, 
        computed during the object deconvolution.
    error_h : array.
        SNR between signal and the auto-correlation of o convolved with h, 
        computed during the PSF deconvolution.

    """
    
    if kerneltype!='A' and kerneltype!='B':
        kerneltype='B'
        print('Wrong input, I have chosen Anchor Update scheme B')
        
    volume = len(signal.shape)>=3
    
    error_o = np.ndarray(0)
    error_h = np.ndarray(0)
    
    if volume == False:
        o = np.zeros([int(iterations_tot/step_image), signal.shape[0], signal.shape[1]])
        h = np.zeros([int(iterations_tot/step_image), signal.shape[0], signal.shape[1]])
    
    for i in np.arange(iterations_tot):
        start_time = time.time()        
        print("\n-----------------------------------------------------")
        print("Blind iteration " + str(i) + " of " + str(iterations_tot))
        if cupy_enabled:
            
            # deconvolution of the object
            if kerneltype=='A' and i==0:
                kernel=H_prior
            elif kerneltype=='A':
                kernel=pf.my_autocorrelation(h_prior)
            elif kerneltype=='B':
                kernel=h_prior
                
            o_prior, error_o_store = pf.anchorUpdateX(cp.asarray(signal), cp.asarray(kernel), signal_deconv=cp.asarray(o_prior), iterations=iterations_o, kerneltype=kerneltype)
        
            # save results
            if (i+1)%step_image == 0 and volume == False:
                o[int(i/step_image),:,:] = o_prior.get()
            error_o = np.append(error_o, error_o_store.get())
            
            # deconvolution of the psf
            if kerneltype=='A':
                kernel=pf.my_autocorrelation(o_prior)
            elif kerneltype=='B':
                kernel=o_prior
            
            h_prior, error_h_store = pf.anchorUpdateX(cp.asarray(signal), cp.asarray(kernel), signal_deconv=cp.asarray(h_prior), iterations=iterations_h, kerneltype=kerneltype)
            
            # save results
            if (i+1)%step_image == 0 and volume == False:
                h[int(i/step_image),:,:] = h_prior.get()
            error_h = np.append(error_h, error_h_store.get())
        
        else:
            
            # deconvolution of the object
            if kerneltype=='A' and i==0:
                kernel=H_prior
            elif kerneltype=='A':
                kernel=pf.my_autocorrelation(h_prior)
            else:
                kernel=h_prior
                
            o_prior, error_o_store = pf.anchorUpdateX(signal, kernel, signal_deconv=o_prior, iterations=iterations_o, kerneltype=kerneltype)
            
            # save results
            if (i+1)%step_image == 0 and volume == False:
                o[int(i/step_image),:,:] = o_prior
            error_o = np.append(error_o, error_o_store)
            
            # deconvolution of the kernel
            if kerneltype=='A':
                kernel=pf.my_autocorrelation(o_prior)
            elif kerneltype=='B':
                kernel=o_prior
            
            h_prior, error_h_store = pf.anchorUpdateX(signal, kernel, signal_deconv=h_prior, iterations=iterations_h, kerneltype=kerneltype)
            
            # save results
            if (i+1)%step_image == 0 and volume == False:
                h[int(i/step_image),:,:] = h_prior
            error_h = np.append(error_h, error_h_store)
        
        print("Performance:")
        print("--- Sample SNR:                   " + str(np.around(error_o[-1], 3)) + " dB")  
        print("--- PSF SNR:                      " + str(np.around(error_h[-1], 3)) + " dB")  
        print("--- Execution time:               %s s" % np.around((time.time() - start_time),3))
        print("--- Execution time per iteration: %s s/step" % np.around(((time.time() - start_time)/(iterations_o + iterations_h)),3)) 
        
    if volume == True:
        o = o_prior
        h = h_prior
        
    return o,h,error_o,error_h


