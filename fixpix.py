# -*- coding: utf-8 -*-
"""
Created on Wed Jul 03 14:35:13 2013

@author: jholt
with minimal modifications and declassification by rcohen
"""
import os
import numpy as np
from numpy import fft
import scipy
from scipy import optimize
import pylab as pl
#import itertools
#from scipy.signal import argrelextrema
#import statsmodels.api as smapi
from statsmodels.formula.api import ols
import config
import matplotlib.pyplot as plt

import logging
from scipy.signal._peak_finding import argrelextrema

logger = logging.getLogger('obj')

#
# function names modified for REDSPEC by SSKIM (Jan 2001)
#


def noise_rs(im):

    # function to find the std dev of a block of pixels

    # sort the array

    imsort = np.sort(im)

    # find the number of pixels

    npix = imsort.shape[0]

    # pick the integer that is closest to 2/3 of the number of pixels
    # this is an integer

    num = 2*npix//3 + 1

    # find the noise as a the minimum difference between
    # the averages of two pairs of points approximately 2/3 of the
    # entire array length apart

    noise = np.min(imsort[num-1:npix-1] + imsort[num-2:npix-2] - imsort[0:npix-num] - imsort[1:npix-num+1]) / 2.0

    # now correct the noise for the fact that it's not exactly over 2/3 of the
    # elements

    noise = noise / (1.9348+((num-2.)/(npix) - 0.66667)/.249891)

    # now correct for the finite number of pixels

    if npix > 100:
        noise = noise / (1.0-0.6141*np.exp(-0.5471*np.log(npix)))
    else:
        noise = noise / (1.0-0.2223*np.exp(-0.3294*np.log(npix)))

    return noise




####################################


def fixpix_rs(im, iternum=3, quiet=True):

    # sigma size problem fixed by SSKIM (Sep 2000)

    # INITIALIZATIONS

    # in case of an error return to the calling prog

    #on_error, 2

    # determine the median value of the entire image

    medimg = np.median(im)

    # create a new image to be the cleaned up output

    imnew = im

    # set the maskarray to zero

    maskarr = im * 0.0

    # set mparm to 1 as a default

    mparm = 1




    # DETERMINE THE AVERAGE NOISE IN THE ENTIRE IMAGE

    # scan through 5x5 boxes to determine some average estimate of the
    # image noise

    xs    = im.shape[0]
    ys    = im.shape[1]
    sigma = np.zeros(xs//5 * ys//5)  # Corrected by SSKIM

    n=0
    for i in range(2, xs-2, 5):

        for j in range(2, ys-2, 5):
            #print(i,j)

            tmp = im[i-2:i+3, j-2:j+3]
            #print(tmp)

            srt = np.sort(tmp).flatten()
            #print(srt)

            #print(srt[16:24])
            #print(srt[15:23])
            #print(srt[0:8])
            #print(srt[1:9])
            #print(srt[16:24] + srt[15:23] - srt[0:8] - srt[1:9])
            #print(np.min(srt[16:24] + srt[15:23] - srt[0:8] - srt[1:9]))
            sigma[n] = (np.min(srt[16:25] + srt[15:24] - srt[0:9] - srt[1:10])/2.0)/1.540

            n += 1
        

    # define the median value of the sigmas, and the sigma of the sigmas

    medsig  = np.median(sigma)
    sigsig  = noise_rs(sigma)  # Modified by SSKIM


    # BEGIN SCANNING FOR HOT & COLD PIXELS

    # start at (4,4) (so that any pixel in the box can have a 5x5 
    # square centered on it.  find the hottest and coldest pixels

    # loop through several iterations of replactments

    for iter1 in range(1, iternum+1):

        hotcnt  = 0                  # variables to count the replacements of hot and cold pixels
        coldcnt = 0
        addon   = ((iter1+1) % 2) * 2

        for i in range(4 + addon, xs-4, 5):

            for j in range(4 + addon, ys-4, 5):

                box     = imnew[i-2:i+3, j-2:j+3].flatten()
                
                hotval  = np.max(box)
                hoti    = np.where(box == hotval)[0][0]

                # coords in original image
                hotx    = hoti % 5 + i-2 
                hoty    = hoti / 5 + j-2

                coldval = np.min(box)
                coldi   = np.where(box == coldval)[0][0]

                # coords in original image
                coldx   = coldi % 5 + i-2
                coldy   = coldi / 5 + j-2

                                    # begin the decision process for the hottest pixel
                hotx    = int(hotx)
                hoty    = int(hoty)
                coldx   = int(coldx)
                coldy   = int(coldy)

                hot     = imnew[hotx-2:hotx+3, hoty-2:hoty+3]

                med8    = np.median(np.append(hot[1:4,1], np.append(hot[1:4,3], [hot[1,2], hot[3,2]])))
                med16   = np.median(np.append(hot[0:5,0], np.append(hot[0:5,4], [np.transpose(hot[0,1:4]), np.transpose(hot[4,1:4])])))
                srt     = np.sort(hot).flatten()
                sig     = (np.min(srt[16:25] + srt[15:24] - srt[0:9] - srt[1:10]) / 2.0) / 1.540

                                    # decide from the noise in the box if we 
                                    # are on a feature or a gaussian background

                if sig > (medsig + 2.0*sigsig):
                    sig = np.max([medsig+2.0*sigsig, np.sqrt(5.0+.210*abs(med8)/mparm)*mparm])

                                    # decide whether to replace pixel

                if ((med8+2.0*med16)/3.0 - medimg) > 2.0*sig:
                    if (imnew[hotx,hoty] > (2.0*med8-medimg+3.0*sig)):
                        imnew[hotx,hoty]   = med8
                        maskarr[hotx,hoty] = 1
                        hotcnt             = hotcnt + 1
                    
                else:
                    if ((imnew[hotx,hoty] - (med8+2.0*med16)/3) > 5.0*sig):
                        imnew[hotx,hoty]   = med8
                        maskarr[hotx,hoty] = 1
                        hotcnt             = hotcnt + 1

                                    # begin the decision process for the coldest pixel
                
                cld   = imnew[coldx-2:coldx+3, coldy-2:coldy+3]
                med8  = np.median(np.append(cld[1:4,1], np.append(cld[1:4,3], [cld[1,2], cld[3,2]])))
                med16 = np.median(np.append(cld[0:5,0], np.append(cld[0:5,4], [np.transpose(cld[0,1:4]), np.transpose(cld[4,1:4])])))
                srt   = np.sort(cld).flatten()
                sig   = (np.min(srt[16:25]+srt[15:24]-srt[0:9]-srt[1:10])/2.0)/1.540

                                    # decide from the noise in the box if we 
                                    # are on a feature or a gaussian background

                if sig > (medsig + 2.0*sigsig):
                  sig = np.max([medsig+2.0*sigsig, np.sqrt(5.0+.210*abs(med8))])

                                    # decide whether to replace pixel

                if ((med8+2.0*med16)/3.0 -medimg) < -2.0*sig:
                    if (imnew[coldx,coldy] < (2.0*med8-medimg-3.0*sig)):
                        imnew[coldx,coldy]   = med8
                        maskarr[coldx,coldy] = 1
                        coldcnt              = coldcnt+1
                    
                else:
                    if ((imnew[coldx,coldy]-(med8+2.0*med16)/3) < -5.0*sig):
                        imnew[coldx,coldy]   = med8
                        maskarr[coldx,coldy] = -1
                        coldcnt              = coldcnt+1
                    

        if not quiet: print('(%s,%s,"  hot and cold pixels in iter. ",%s)'%(hotcnt,coldcnt,iter1))

    return imnew
