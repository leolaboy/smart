import numpy as np
from scipy.signal._peak_finding import argrelextrema

import logging
import config

import tracer
import matplotlib.pyplot as plt

logger = logging.getLogger('obj')
      
        
def calc_noise_img(obj, flat, integration_time):
    """
    flat is expected to be normalized and both obj and flat are expected to be rectified
    """
    
    G  = 5.8  # e-/ADU    
    RN = 23.0 # e-
    DC = 0.8  # e-/second/pixel
    
    # calculate photon noise
    #noise = obj / G # What is this?
    noise = obj # This is ALREADY in ADU!
    #noise = obj * G # This is in electrons
    
    # add read noise
    noise += np.square(RN / G) # This is in ADU
    #noise += np.square(RN) # This is in electrons
    
    # add dark current noise
    noise += (DC / G) * integration_time # This is in ADU
    #noise += (DC / G) * integration_time # This is in electrons
    
    # divide by normalized flat squared
    noise /= np.square(flat)
    
    return noise


ORDER_EDGE_SEARCH_WIDTH = 10
ORDER_EDGE_BG_WIDTH     = 30
ORDER_EDGE_JUMP_THRESH  = 1.9
ORDER_EDGE_JUMP_LIMIT   = 200

def trace_order_edge(data, start):
        
    trace, nJumps =  tracer.trace_edge(
            data, start, ORDER_EDGE_SEARCH_WIDTH, ORDER_EDGE_BG_WIDTH, ORDER_EDGE_JUMP_THRESH)
    
    if trace is None:
        logger.warning('trace failed')
        return None
    
#     if slit_name.endswith('24'):
#         x = np.arange(len(trace))
#         coeffs = np.polyfit(x, trace, 1)
#         y_fit = np.polyval(coeffs, x)
#         res1 = trace - y_fit
# #         stdev1 = np.std(res1)
# 
#         if abs(coeffs[0]) < 1e-2:
#             logger.warning('long slit edge criteria not met')
#             return None
#         else:
#             return trace
    
    if nJumps > ORDER_EDGE_JUMP_LIMIT:
        
            logger.debug('order edge trace jump limit exceeded')
            logger.debug('reducing search width to {:.1f}'.format(ORDER_EDGE_SEARCH_WIDTH / 1.5))
            trace, nJumps =  tracer.trace_edge(
            data, start, ORDER_EDGE_SEARCH_WIDTH / 2, ORDER_EDGE_BG_WIDTH, ORDER_EDGE_JUMP_THRESH)
            
            if trace is None:
                logger.warning('trace failed')
                return None
            
            if nJumps > ORDER_EDGE_JUMP_LIMIT:
                logger.info('order edge trace jump limit exceeded: n jumps=' + 
                        str(nJumps) + ' limit=' + str(ORDER_EDGE_JUMP_LIMIT))
                if config.params['spatial_jump_override'] is True:
                    logger.warning('spatial jump override enabled, edge not rejected')
                else:
                    logger.info('edge rejected')
                    return None
    return trace
    
SKY_LINE_SEARCH_WIDTH = 3
SKY_LINE_BG_WIDTH     = 0
SKY_LINE_JUMP_THRESH  = 0.8
SKY_LINE_JUMP_LIMIT   = 10
        
def trace_sky_line(data, start, eta=None):

    if eta is not None: # Change the parameters a little for Etalon lamps
        SKY_LINE_JUMP_THRESH = 1
        SKY_LINE_SEARCH_WIDTH = 10

    trace, nJumps =  tracer.trace_edge(
            data, start, SKY_LINE_SEARCH_WIDTH, SKY_LINE_BG_WIDTH, SKY_LINE_JUMP_THRESH, eta=eta)

    if trace is None:
        logger.warning('sky line trace failed')
        return None
    if nJumps > SKY_LINE_JUMP_LIMIT:
        logger.debug('sky line trace jump limit exceeded: n jumps=' + 
                str(nJumps) + ' limit=' + str(SKY_LINE_JUMP_LIMIT))        
        return None
    return trace



def smooth_spatial_trace(y_raw):
    """
    """
    
    deg = 3
    n_end_ignore = 20
    threshold = 3
    
    mask = np.ones(y_raw.shape[0] - n_end_ignore, dtype=bool)
    mask = np.append(mask, np.zeros(n_end_ignore, dtype=bool))
    
    x = np.arange(y_raw.shape[0])
    
    coeffs = np.polyfit(x[mask], y_raw[mask], deg)
    y_fit = np.polyval(coeffs, x)
    res1 = y_raw - y_fit
    stdev1 = np.std(res1)
    
    greater = np.greater(np.absolute(res1), threshold * stdev1)
    mask = np.logical_and(mask, np.logical_not(greater))
    
    coeffs = np.polyfit(x[mask], y_raw[mask], deg)
    y_fit = np.polyval(coeffs, x)
    res2 = y_raw - y_fit
    stdev2 = np.std(res2)
 
    return y_fit, mask

SKY_SIGMA           = 2.25
EXTRA_PADDING       = 5
MIN_LINE_SEPARATION = 5



def find_spectral_trace(data, numrows=5, eta=None, TEST=False):
    """
    Locates sky lines in the bottom 5 rows (is this really optimal?) of the order image data. 
    Finds strongest peaks, sorts them, traces them, returns the average of the traces.
    Rejects line pairs that are too close together.
    Returns spectral trace as 1-d array.  Throws exception if can't find or trace lines.
    """
    
    # transpose the array because spectroid can only read horizontal peaks for now
    data_t = data.transpose()

#     data_t = data_t[:, padding + 5:data_t.shape[1] - 5 - padding]
    data_t = data_t[:, 5:data_t.shape[1] - 5]
    s = np.sum(data_t[:, 0:numrows], axis=1)

    #import matplotlib.pyplot as plt
    #plt.figure(10)
    #plt.imshow(data_t)#, origin='lower')
    
    # import pylab as pl
    # pl.figure(facecolor='white')
    # pl.cla()
    # pl.plot(s, 'k-')
    # pl.xlim(0, 1024)
    # pl.xlabel('column (pixels)')
    # pl.ylabel('intensity summed over 5 rows (DN)')
    # pl.show()

    # finds column indices of maxima
    if eta is not None:
        maxima_c = argrelextrema(s, np.greater, order=3) 
    else:
        maxima_c = argrelextrema(s, np.greater)    
    
    # find indices in maxima_c of maxima with intensity 
    # greater than SKY_SIGMA * mean extrema height
    if eta is not None:# Do it slightly different for the etalon lamps
        sky_thres = 1.2 * np.median(s)
    else:
        sky_thres = SKY_SIGMA * s.mean()
    locmaxes = np.where(s[maxima_c[0]] > sky_thres)
    
    # indices in s or peaks
    maxes = np.array(maxima_c[0][locmaxes[0]])
    
    logger.info('n sky/etalon line peaks with intensity > {:.0f} = {}'.format(
            sky_thres, len(maxes)))

    deletelist = []
   
    # remove adjacent sky lines that are closer than MIN_LINE_SEPARATION pixels
    for i in range(1, len(maxes)):
        if abs(maxes[i] - maxes[i - 1]) < MIN_LINE_SEPARATION:
            deletelist.append(i)
    maxes = np.delete(maxes, deletelist, None)

    peaks = s[maxes] 

    sortorder = np.argsort(peaks)
            
    maxes = maxes[sortorder]
    maxes = maxes[::-1]

    centroid_sky_sum = np.zeros(data_t.shape[1])
    fitnumber = 0

    #print('MAXES', maxes)
    #centroids = np.array([])
    #Pixels    = np.array([])
    centroids = []
    for maxskyloc in maxes:
        #print('MAX LOC', maxskyloc)
        if 10 < maxskyloc < 1010:
            
            centroid_sky = trace_sky_line(data_t, maxskyloc, eta=eta)
           
            if centroid_sky is None:
                continue

            fitnumber += 1
            centroid_sky_sum = centroid_sky_sum + centroid_sky - centroid_sky[0]
            if eta is not None:
                #centroids = np.concatenate((centroids, centroid_sky - centroid_sky[0]))
                #Pixels    = np.concatenate((Pixels, np.arange(len(centroid_sky))))
                centroids.append(centroid_sky - centroid_sky[0])
            #print('SUM0', centroid_sky - centroid_sky[0])
            #print('SUM1', centroids)

            if eta is None:
                if fitnumber > 2: # Why are we limiting this?
                    break

            #p0    = np.polyfit(np.arange(len(centroid_sky)), centroid_sky, deg=1) 
            #z0    = np.poly1d(p0)
            #plt.scatter(np.arange(len(centroid_sky)), centroid_sky, color='r', s=1, alpha=0.5)
            #plt.plot(np.arange(len(centroid_sky)), z0(np.arange(len(centroid_sky))), 'r--', lw=0.5)

    
    #print('SUM2', centroid_sky_sum)
    #plt.show()
    #sys.exit()

    if centroid_sky_sum.any():
        if eta is not None:
            logger.info(str(fitnumber) + ' etalon lines used for spectral rectification')
            #return [Pixels, centroids]
            if TEST == True: return centroids
            else: return centroid_sky_sum / fitnumber
        else:
            logger.info(str(fitnumber) + ' sky lines used for spectral rectification')
            return centroid_sky_sum / fitnumber
    
    logger.warning('failed to find sky/etalon line trace')
    raise StandardError('failed to find sky/etalon line trace')
    
    
def smooth_spectral_trace(data, l, eta=None, TEST=False):
    
    if TEST == True:
        if eta is not None:
            #print('DATA:', data)
            #Pixels, centroids = data[0], data[1]
            for centroids in data:
                Pixels = np.arange(len(centroids))
                p0 = np.polyfit(Pixels, centroids, deg=1)  # end always drops off
                """
                plt.figure(161)
                z0 = np.poly1d(p0)
                plt.scatter(Pixels, centroids)
                Xs = np.arange(np.min(Pixels), np.max(Pixels))
                plt.plot(Xs, z0(Xs), 'r--')
                plt.figure(162)
                plt.plot(Pixels, centroids - z0(Pixels))
                plt.axhline(0, c='r', ls='--')
                """
                from astropy.stats import sigma_clip
                SigCut = True
                newData = centroids
                newPix2  = Pixels
                print('1', len(newData), len(newPix2))
                while SigCut:

                    p0 = np.polyfit(newPix2, newData, deg=1)  # end always drops off
                    z0 = np.poly1d(p0)

                    filtered_data = sigma_clip(newData - z0(newPix2), sigma=3, iters=None)

                    newCent = newData[~filtered_data.mask]
                    newPix  = newPix2[~filtered_data.mask]

                    #plt.figure(163)
                    #plt.scatter(newPix, newCent)
                    #plt.plot(Xs, z0(Xs), 'r--')
                    #plt.figure(164)
                    #plt.plot(newPix, newCent - z0(newPix))
                    #plt.axhline(0, c='r', ls='--')
                    #plt.show()
                    if len(newData) == len(newCent): 
                        SigCut = False
                    newData = newCent
                    newPix2 = newPix
                    print(len(newData), len(newPix2))
                print('3) spectral tilt is {:.3f} pixels/pixel'.format(p0[0]))
        
            
            """
            p0 = np.polyfit(Pixels, centroids, deg=1)  # end always drops off
            plt.figure(161)
            z0 = np.poly1d(p0)
            plt.scatter(Pixels, centroids)
            Xs = np.arange(np.min(Pixels), np.max(Pixels))
            plt.plot(Xs, z0(Xs), 'r--')
            plt.show()
            sys.exit()
            """
            logger.info('TEST spectral tilt is {:.3f} pixels/pixel'.format(p0[0]))
            fit = np.polyval(p0, np.arange(l))
            #print('fit', fit)
            #return fit

        else:

            p0 = np.polyfit(np.arange(len(data) - 10), data[:-10], deg=1)  # end always drops off
            plt.figure(161)
            z0 = np.poly1d(p0)
            plt.scatter(np.arange(len(data) - 10), data[:-10])
            plt.plot(np.arange(len(data) - 10), z0(np.arange(len(data) - 10)), 'r--')
            plt.show()
            sys.exit()
            #p0 = np.polyfit(np.arange(len(data)), data, deg=1)  # end always drops off, but this doesn't care
            logger.info('spectral tilt is {:.3f} pixels/pixel'.format(p0[0]))
            fit = np.polyval(p0, np.arange(l))
    
    '''
    p0 = np.polyfit(np.arange(len(data) - 10), data[:-10], deg=1)  # end always drops off
    #plt.figure(161)
    #z0 = np.poly1d(p0)
    #plt.scatter(np.arange(len(data) - 10), data[:-10])
    #plt.plot(np.arange(len(data) - 10), z0(np.arange(len(data) - 10)), 'r--')
    #plt.show()
    #sys.exit()
    #p0 = np.polyfit(np.arange(len(data)), data, deg=1)  # end always drops off, but this doesn't care
    '''              

    from astropy.stats import sigma_clip
    SigCut   = True
    newData  = data[:-10]
    newPix2  = np.arange(len(data)-10)
    Xs       = np.arange(np.min(newPix2), np.max(newPix2))
    while SigCut:

        p0 = np.polyfit(newPix2, newData, deg=1)  # end always drops off
        z0 = np.poly1d(p0)

        filtered_data = sigma_clip(newData - z0(newPix2), sigma=3, iters=None)

        newCent = newData[~filtered_data.mask]
        newPix  = newPix2[~filtered_data.mask]
        """
        plt.figure(163)
        plt.scatter(newPix2, newData, alpha=0.5, c='r', s=5)
        plt.scatter(newPix, newCent, alpha=0.5, c='b', s=5)
        plt.plot(Xs, z0(Xs), 'r--')
        plt.figure(164)
        plt.plot(newPix, newCent - z0(newPix))
        plt.axhline(0, c='r', ls='--')
        plt.show()
        """
        if len(newData) == len(newCent): 
            SigCut = False

        newData = newCent
        newPix2 = newPix
        #print(len(newData), len(newPix2))
    
    #print('3.1) spectral tilt is {:.3f} pixels/pixel'.format(p0[0]))
    
    logger.info('spectral tilt is {:.3f} pixels/pixel'.format(p0[0]))
    fit = np.polyval(p0, np.arange(l))
    return fit