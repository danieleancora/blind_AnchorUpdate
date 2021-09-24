# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 09:30:52 2021
@author: Elena Corbetta, Daniele Ancora

These functions are taken from PYPHRET package from the link:
https://github.com/danieleancora/pyphret

"""

import time
import numpy as np

######### import cupy only if installed #########
from importlib import util
cupy_enabled = util.find_spec("cupy") is not None
if cupy_enabled:
    import cupy  as cp
######### ----------------------------- #########


#%% From pyphret.functions:
#   https://github.com/danieleancora/pyphret/blob/master/functions.py
    
# flip all axis, it should return a view not a new vector. I need to write this because cupy flip does not work the same way as numpy
def axisflip(kernel):
    if kernel.ndim==1:
        kernel_flip = kernel[::-1]
    elif kernel.ndim==2:
        kernel_flip = kernel[::-1,::-1]
    elif kernel.ndim==3:
        kernel_flip = kernel[::-1,::-1,::-1]
    elif kernel.ndim==4:
        kernel_flip = kernel[::-1,::-1,::-1,::-1]

    return kernel_flip


def my_convolution(function1, function2):
    xp = get_array_module(function1)
    # return xp.fft.fftshift(xp.fft.irfftn(xp.fft.rfftn(function1) * xp.fft.rfftn(function2), s=function1.shape))
    return xp.fft.ifftshift(xp.fft.irfftn(xp.fft.rfftn(function1) * xp.fft.rfftn(function2), s=function1.shape))


def my_correlation(function1, function2):
    xp = get_array_module(function1)
    return xp.fft.ifftshift(xp.fft.irfftn(xp.conj(xp.fft.rfftn(function1)) * xp.fft.rfftn(function2), s=function1.shape))


def my_autocorrelation(x):
    return my_correlation(x, x)


def my_convcorr_sqfft(function1, function2):
    xp = get_array_module(function1)
    temp = xp.conj(xp.fft.rfftn(function1)) * function2
    return xp.fft.irfftn(temp, s=function1.shape)


def gaussian_psf(size=[200,200], alpha=[10,20]):
    x = np.arange(0, size[0], dtype=np.float32)
    gaussian_1d = ( np.exp(-(x - size[0] / 2 + 0.5)**2.0 / (2 * alpha[0]**2)) ) 
    dim = len(alpha)
    # we use outer products to compute the psf accordingly to given dimensions
    if dim == 1:
        psf = gaussian_1d        
    if dim == 2:
        x = np.arange(0, size[1], dtype=np.float32)
        gaussian_2d = ( np.exp(-(x - size[1] / 2 + 0.5)**2.0 / (2 * alpha[1]**2)) ) 
        psf =  gaussian_1d.reshape(gaussian_1d.shape[0],1)*gaussian_2d        
    elif dim == 3:
        x = np.arange(0, size[1], dtype=np.float32)
        gaussian_2d = ( np.exp(-(x - size[1] / 2 + 0.5)**2.0 / (2 * alpha[1]**2)) ) 
        x = np.arange(0, size[2], dtype=np.float32)
        gaussian_3d = ( np.exp(-(x - size[2] / 2 + 0.5)**2.0 / (2 * alpha[2]**2)) ) 
        psf = (gaussian_1d.reshape(gaussian_1d.shape[0],1)*gaussian_2d).reshape(gaussian_1d.shape[0],gaussian_2d.shape[0],1)*gaussian_3d

    return psf    


# signal-to-noise ratio definition
def snrIntensity_db(signal, noise, kind='mean'):
    xp = get_array_module(signal)
    if kind=='mean':
        return 20*xp.log10(xp.mean(signal) / xp.mean(noise))
    if kind=='peak':
        return 20*xp.log10(xp.max(signal) / xp.mean(noise))


#%% From pyphret.deconvolutions
#   https://github.com/danieleancora/pyphret/blob/master/deconvolutions.py

def anchorUpdateX(signal, kernel, signal_deconv=np.float32(0), kerneltype = 'B', iterations=10, measure=True, clip=False, verbose=True):
    """
    Reconstruction of signal_deconv from its auto-correlation signal, via a 
    RichardsonLucy-like multiplicative procedure. At the same time, the kernel 
    psf is deconvolved from the reconstruction so that the iteration converges
    corr(conv(signal_deconv, kernel), conv(signal_deconv, kernel),) -> signal.

    Parameters
    ----------
    signal : ndarray, either numpy or cupy. 
        The auto-correlation to be inverted
    kernel : ndarray, either numpy or cupy.
        Point spread function that blurred the signal. It must be 
        signal.shape == kernel.shape.
    signal_deconv : ndarray, either numpy or cupy or 0. It must be signal.shape == signal_deconv.shape.
        The de-autocorrelated signal deconvolved with kernel at ith iteration. The default is np.float32(0).
    kerneltype : string.
        Type of kernel update used for the computation choosing from blurring 
        directly the autocorrelation 'A', blurring the signal that is then 
        autocorrelated 'B' and the window applied in fourier domain 'C'. 
        The default is 'B'.
    iterations : int, optional
        Number of iteration to be done. The default is 10.
    measure : boolean, optional
        If true computes the euclidean distance between signal and the 
        auto-correlation of signal_deconv. The default is True.
    clip : boolean, optional
        Clip the results within the range -1 to 1. Useless for the moment. The default is False.
    verbose : boolean, optional
        Print current step value. The default is True.

    Returns
    -------
    signal_deconv : ndarray, either numpy or cupy.
        The de-autocorrelated signal deconvolved with kernel at ith iteration..
    error : vector.
        Euclidean distance between signal and the auto-correlation of signal_deconv.
        Last implementation returns the SNR instead of euclidean distance.

    """
    
    # for code agnosticity between Numpy/Cupy
    xp = get_array_module(signal)
    
    # for performance evaluation
    start_time = time.time()
    
    if iterations<100: 
        breakcheck = iterations
    else:
        breakcheck = 100

    # normalization
    signal /= signal.sum()
    kernel /= kernel.sum()
    epsilon = 1e-7

    # compute the norm of the fourier transform of the kernel associated with the IEEE paper
    if kerneltype == 'A':
        kernel = xp.abs(xp.fft.rfftn(kernel))
    elif kerneltype == 'B':
        kernel = xp.square(xp.abs(xp.fft.rfftn(kernel)))
    elif kerneltype == 'C':
        kernel = xp.abs(xp.fft.irfftn(kernel))
    else:
        print('Wrong input, I have choosen Anchor Update scheme, B')
        kernel = xp.square(xp.abs(xp.fft.rfftn(kernel)))

    # starting guess with a flat image
    if signal_deconv.any()==0:
        # xp.random.seed(0)
        signal_deconv = xp.full(signal.shape,0.5) + 0.01*xp.random.rand(*signal.shape)
        # signal_deconv = signal.copy()
    else:
        signal_deconv = signal_deconv #+ 0.1*prior.max()*xp.random.rand(*signal.shape)
    
    # normalization
    signal_deconv = signal_deconv/signal_deconv.sum()
        
    # to measure the distance between the guess convolved and the signal
    error = None    
    if measure == True:
        error = xp.zeros(iterations)

    for i in range(iterations):
        # I use this property to make computation faster
        kernel_update = my_convcorr_sqfft(signal_deconv, kernel)
        kernel_mirror = axisflip(kernel_update)
        
        relative_blur = my_convolution(signal_deconv, kernel_update)
        # relative_blur = pyconv.convolve(signal_deconv, kernel_update, mode='same', method='fft')
        
        # compute the measured distance metric if given
        if measure==True:
            # error[i] = xp.linalg.norm(signal/signal.sum()-relative_blur/relative_blur.sum())
            error[i] = snrIntensity_db(signal/signal.sum(), xp.abs(signal/signal.sum()-relative_blur/relative_blur.sum()))
            if (error[i] < error[i-breakcheck]) and i > breakcheck:
                break

        if verbose==True and (i % 100)==0 and measure==False:
            pass
            print('Iteration ' + str(i))
        elif verbose==True and (i % 100)==0 and measure==True:
            pass
            #print('--- Signal to noise ratio: ' + str(np.around(error[i], 3)) + ' dB')

        relative_blur = signal / relative_blur

        # avoid errors due to division by zero or inf
        relative_blur[xp.isinf(relative_blur)] = epsilon
        relative_blur = xp.nan_to_num(relative_blur)

        # multiplicative update, for the full model
        # signal_deconv *= 0.5 * (my_convolution(relative_blur, kernel_mirror) + my_correlation(axisflip(relative_blur), kernel_mirror))
        # signal_deconv *= (my_convolution(relative_blur, kernel_mirror) + my_correlation(relative_blur,kernel_mirror))


        # multiplicative update, for the Anchor Update approximation
        signal_deconv *= my_convolution(relative_blur, kernel_mirror)

        # multiplicative update, remaining term. This gives wrong reconstructions
        # signal_deconv *= my_correlation(axisflip(relative_blur), kernel_mirror)
                
    if clip:
        signal_deconv[signal_deconv > +1] = +1
        signal_deconv[signal_deconv < -1] = -1

    # print("\n Performance:")
    # print("--- %s s" % (time.time() - start_time))
    # print("--- %s s/step" % ((time.time() - start_time)/iterations))
    return signal_deconv, error #,kernel_update
    # return kernel_mirror, error #



# %% From pyphret.backend
#    https://github.com/danieleancora/pyphret/blob/master/backend.py


# inspired by sigpy
# https://github.com/mikgroup/sigpy/blob/master/sigpy/config.py

# this function is taken from SIGPY package from the link
# https://github.com/mikgroup/sigpy/blob/5bd25cdfda5b72c2728993ad5e6f7288f274ddc4/sigpy/backend.py
def get_array_module(array):
    """Gets an appropriate module from :mod:`numpy` or :mod:`cupy`.

    This is almost equivalent to :func:`cupy.get_array_module` and here it is 
    ment to replace it. The difference is that this function can be used even 
    if cupy is not available.

    Args:
        array: Input array.

    Returns:
        module: :mod:`cupy` or :mod:`numpy` is returned based on input.
    """
    if cupy_enabled:
        return cp.get_array_module(array)
    else:
        return np



