import logging
import numpy as np
import scipy.stats
import scipy.optimize
#import scipy.ndimage

import config
import image_lib
import nirspec_lib
import wavelength_utils
import Line
import DrpException

from astropy.visualization import ZScaleInterval, ImageNormalize
import fixpix

import CAT_Functions as cat
import matplotlib.pyplot as plt 

logger = logging.getLogger('obj')

def reduce_order(order, eta=None):
        
    #print('ETA BEGINNING', eta)
    #print(order.flatOrder.orderNum)
    #if order.flatOrder.orderNum != 33: continue
    #sys.exit()

    # flatten object images for this order
    __flatten(order, eta=eta)

    """
    from astropy.visualization import ZScaleInterval, ImageNormalize
    plt.figure(1)
    norm = ImageNormalize(order.ffEtaImg, interval=ZScaleInterval())
    plt.imshow(order.ffEtaImg, origin='lower')
    plt.show()
    sys.exit()
    """

    ### XXX TESTING AREA
    
    if config.params['no_clean']:
        logger.info("bad pixel rejection on object frame/order inhibited by command line flag")

    else:
        for frame in order.frames:
            if frame == 'AB': continue # Don't need to do this one
            logger.info('bad pixel cleaning object frame %s'%frame)
            order.ffObjImg[frame] = fixpix.fixpix_rs(order.ffObjImg[frame])
            logger.debug('bad pixel cleaning object frame %s complete'%frame)
        
        if eta is not None:
            logger.info('bad pixel cleaning etalon frame')
            order.ffEtaImg = fixpix.fixpix_rs(order.ffEtaImg)
            logger.debug('bad pixel cleaning etalon frame complete')
    
    ### XXX TESTING AREA

    """
    plt.figure(3)
    norm = ImageNormalize(order.ffEtaImg, interval=ZScaleInterval())
    plt.imshow(order.ffEtaImg, origin='lower', norm=norm, aspect='auto')
    #np.save('unrect_%s.npy'%order.flatOrder.orderNum, order.ffEtaImg) 
    #plt.savefig('%s_unrect.png'%order.flatOrder.orderNum, dpi=600, bbox_inches='tight')
    plt.figure(33)
    norm = ImageNormalize(order.ffObjImg['A'], interval=ZScaleInterval())
    plt.imshow(order.ffObjImg['A'], origin='lower', aspect='auto', norm=norm)
    #np.save('unrect_%s.npy'%order.flatOrder.orderNum, order.ffEtaImg) 
    #plt.savefig('%s_unrect.png'%order.flatOrder.orderNum, dpi=600, bbox_inches='tight')
    plt.figure(333)
    norm = ImageNormalize(order.ffObjImg['B'], interval=ZScaleInterval())
    plt.imshow(order.ffObjImg['B'], origin='lower', aspect='auto', norm=norm)
    plt.show(block=True)
    """
 
    # rectify obj and flattened obj in spatial dimension
    __rectify_spatial(order, eta=eta)
 
    # trim rectified order
    __trim(order, eta=eta)

    """
    plt.figure(4)
    norm = ImageNormalize(order.ffEtaImg, interval=ZScaleInterval())
    plt.imshow(order.ffEtaImg, origin='lower', norm=norm, aspect='auto')
    #np.save('unrect_%s.npy'%order.flatOrder.orderNum, order.ffEtaImg) 
    #plt.savefig('%s_unrect.png'%order.flatOrder.orderNum, dpi=600, bbox_inches='tight')
    plt.figure(44)
    norm = ImageNormalize(order.ffObjImg['A'], interval=ZScaleInterval())
    plt.imshow(order.ffObjImg['A'], origin='lower', aspect='auto', norm=norm)
    #np.save('unrect_%s.npy'%order.flatOrder.orderNum, order.ffEtaImg) 
    #plt.savefig('%s_unrect.png'%order.flatOrder.orderNum, dpi=600, bbox_inches='tight')
    plt.figure(444)
    norm = ImageNormalize(order.ffObjImg['B'], interval=ZScaleInterval())
    plt.imshow(order.ffObjImg['B'], origin='lower', aspect='auto', norm=norm)
    plt.show(block=True)
    """

    # save spatially rectified images before spectral rectify for diagnostics 
    # if AB pair then subtract B from A
    if order.isPair:
        order.objImg['AB']   = np.subtract(order.objImg['A'], order.objImg['B'])
        order.ffObjImg['AB'] = np.subtract(order.ffObjImg['A'], order.ffObjImg['B'])
        # reFlatten
        if np.amin(order.ffObjImg['AB']) < 0: order.ffObjImg['AB'] -= np.amin(order.ffObjImg['AB'])
    order.srNormFlatImg = order.flatOrder.rectFlatImg
    for frame in order.frames:
        order.srFfObjImg[frame] = order.ffObjImg[frame]

        if eta is not None:
            if frame == 'B':
                order.srNormEtaImgB = order.ffEtaImgB
            else:
                order.srNormEtaImg  = order.ffEtaImg
    
    # find spatial profile and peak
    __find_spatial_profile_and_peak(order)
    
    # characterize spatial profile by fitting to Gaussian
    __characterize_spatial_profile(order)

    # Try to find the spectral trace using etalon lamps if provided
    for frame in order.frames:
        #print('FRAME', frame)
        if frame == 'AB': continue # Build the AB frame using A and B later
        if frame in ['A']:
            if eta is not None: # Do spectral rectification using the etalon lamps
                try:           
                    logger.info('attempting rectification of frame {} order {} in spectral dimension (etalon)'.format(
                                frame, order.flatOrder.orderNum))
                    order.spectralTrace[frame] = nirspec_lib.smooth_spectral_trace(
                                                 nirspec_lib.find_spectral_trace(order.ffEtaImg, eta=eta), 
                                                 order.ffObjImg['A'].shape[0])
                except Exception as e:
                    logger.warning('not rectifying frame {} order {} in spectral dimension (etalon)'.format(
                                   frame, order.flatOrder.orderNum))

            else: # no etalons
                try:
                    logger.info('attempting rectification of frame {} order {} in spectral dimension'.format(
                                frame, order.flatOrder.orderNum))
                    order.spectralTrace[frame] = nirspec_lib.smooth_spectral_trace(
                                                 nirspec_lib.find_spectral_trace(order.ffObjImg['A']), 
                                                 order.ffObjImg['A'].shape[0])
                except Exception as e:
                    logger.warning('not rectifying frame {} order {} in spectral dimension'.format(
                                   frame, order.flatOrder.orderNum))

        else: # B frame
            if eta is not None: # Do spectral rectification using the etalon lamps
                try:           
                    logger.info('attempting rectification of frame {} order {} in spectral dimension (etalon)'.format(
                                frame, order.flatOrder.orderNum))
                    order.spectralTrace[frame] = nirspec_lib.smooth_spectral_trace(
                                                 nirspec_lib.find_spectral_trace(order.ffEtaImgB, eta=eta), 
                                                 order.ffObjImg['B'].shape[0])
                except Exception as e:
                    logger.warning('not rectifying frame {} order {} in spectral dimension (etalon)'.format(
                                   frame, order.flatOrder.orderNum))
            else: # no etalons
                try:
                    logger.info('attempting rectification of frame {} order {} in spectral dimension'.format(
                        frame, order.flatOrder.orderNum))
                    order.spectralTrace[frame] = nirspec_lib.smooth_spectral_trace(
                            nirspec_lib.find_spectral_trace(
                                    order.ffObjImg['B']), order.ffObjImg['B'].shape[0])
                except Exception as e:
                    logger.warning('not rectifying frame {} order {} in spectral dimension'.format(
                                   frame, order.flatOrder.orderNum))
    try: 
        __rectify_spectral(order, eta=eta)
    except:
        logger.warning('not able to rectify all of order {} in spectral dimension'.format(
                                   order.flatOrder.orderNum))

    # if AB pair then subtract B from A
    if order.isPair:
        order.objImg['AB']   = np.subtract(order.objImg['A'], order.objImg['B'])
        order.ffObjImg['AB'] = np.subtract(order.ffObjImg['A'], order.ffObjImg['B'])
        # reFlatten
        if np.amin(order.ffObjImg['AB']) < 0: order.ffObjImg['AB'] -= np.amin(order.ffObjImg['AB'])
    

    #plt.figure(666) #XXX
    #plt.imshow(order.ffEtaImg, origin='lower')
    #np.save('rect_%s.npy'%order.flatOrder.orderNum, order.ffEtaImg) 
    #plt.savefig('%s_rect.png'%order.flatOrder.orderNum, dpi=600, bbox_inches='tight')
    #plt.show()
    

    # compute noise image
    for frame in order.frames:
        order.noiseImg[frame] = nirspec_lib.calc_noise_img(
                order.objImg[frame], order.flatOrder.rectFlatImg, order.integrationTime)

    # extract spectra
    __extract_spectra(order, eta=eta)
            
    # calculate approximate SNR
    __calc_approximate_snr(order)
            
    # find and identify sky/etalon lines   
    line_pairs = None # line_pairs are (column number, accepted wavelength)
    try:
        if eta is not None:
            etalon_wavelengths, etalon_intensities = wavelength_utils.get_etalon_lines()
        else:
            oh_wavelengths, oh_intensities = wavelength_utils.get_oh_lines()
    except IOError as e:
        logger.critical('cannot read OH/Etalon line file: ' + str(e))
        raise
        

    try:
        # synthesize sky/etalon spectrum and store in order object
        if eta is not None:
            order.synthesizedSkySpec = wavelength_utils.synthesize_sky(
                    etalon_wavelengths, etalon_intensities, order.flatOrder.gratingEqWaveScale, eta=eta)
         
            # identify lines and return list of (column number, accepted wavelength) tuples
            line_pairs = wavelength_utils.line_id(order, etalon_wavelengths, etalon_intensities, eta=eta)

        else:
            order.synthesizedSkySpec = wavelength_utils.synthesize_sky(
                    oh_wavelengths, oh_intensities, order.flatOrder.gratingEqWaveScale)
         
            # identify lines and return list of (column number, accepted wavelength) tuples
            line_pairs = wavelength_utils.line_id(order, oh_wavelengths, oh_intensities)
        
    except (IOError, ValueError) as e:
        logger.warning('sky/etalon line matching failed: ' + str(e))
        
    if line_pairs is not None:
        
        logger.info(str(len(line_pairs)) + ' matched sky/etalon lines found in order')

        # add line pairs to Order object as Line objects
        for line_pair in line_pairs:
            col, waveAccepted = line_pair
            peak = order.skySpec['A'][col]
            cent = image_lib.centroid(order.skySpec['A'], 1024, 5, col)
            line = Line.Line(order.baseNames['A'], order.flatOrder.orderNum, 
                    waveAccepted, col, cent, peak)
            order.lines.append(line)
            
        if len(order.lines) >= 3:
            # do local wavelength fit
            measured = []
            accepted = []
            for line in order.lines:
                measured.append(order.flatOrder.gratingEqWaveScale[line.col])
                accepted.append(line.waveAccepted)
            (order.orderCalSlope, order.orderCalIncpt, order.orderCalCorrCoeff, p, e) = \
                    scipy.stats.linregress(np.array(measured), np.array(accepted))  
            order.orderCalNLines = len(order.lines)       
            logger.info('per order wavelength fit: n = {}, a = {:.6f}, b = {:.6f}, r = {:.6f}'.format(
                    len(order.lines), order.orderCalIncpt, order.orderCalSlope, 
                    order.orderCalCorrCoeff))

            for line in order.lines:
                line.orderWaveFit = order.orderCalIncpt + \
                    (order.orderCalSlope * order.flatOrder.gratingEqWaveScale[line.col])    
                line.orderFitRes = abs(line.orderWaveFit - line.waveAccepted)  
                line.orderFitSlope = (order.orderCalSlope * 
                        (order.flatOrder.gratingEqWaveScale[1023] - 
                         order.flatOrder.gratingEqWaveScale[0]))/1024.0
    else:
        logger.warning('no matched sky/etalon lines in order ' + str(order.flatOrder.orderNum))
        order.orderCal = False 

    #plt.show()
                        
    return



def __flatten(order, eta=None):
    """Flat field object image[s] but keep originals for noise calculation.
    """
    
    for frame in order.frames:
        
        order.objImg[frame]   = np.array(order.objCutout[frame]) 
        order.ffObjImg[frame] = np.array(order.objCutout[frame] / order.flatOrder.normFlatImg)
        
        if frame != 'AB':
            if np.amin(order.ffObjImg[frame]) < 0:
                order.ffObjImg[frame] -= np.amin(order.ffObjImg[frame])

        if eta is not None:
            if frame == 'B':
                order.etaImgB   = np.array(order.etaCutout) 
                order.ffEtaImgB = np.array(order.etaCutout / order.flatOrder.normFlatImg)
            else:
                order.etaImg   = np.array(order.etaCutout) 
                order.ffEtaImg = np.array(order.etaCutout / order.flatOrder.normFlatImg)

    
    order.flattened = True
    logger.info('order has been flat fielded')
    return



def __rectify_spatial(order, eta=None):
    """
    """     
    
    for frame in order.frames:
        if frame == 'AB': continue # Skip the AB frame, we will subtract them after rectification

        logger.info('attempting spatial rectification using object trace')
        try:
            if frame in ['A', 'B']:
                #print('FRAME', frame)
                if config.params['onoff'] == True and frame == 'B': 
                    order.objImg[frame]   = image_lib.rectify_spatial(order.objImg[frame], polyVals1)
                    order.ffObjImg[frame] = image_lib.rectify_spatial(order.ffObjImg[frame], polyVals2)

                else:
                    polyVals1             = cat.CreateSpatialMap(order.objImg[frame])  
                    order.objImg[frame]   = image_lib.rectify_spatial(order.objImg[frame], polyVals1)
                    polyVals2             = cat.CreateSpatialMap(order.ffObjImg[frame])  
                    order.ffObjImg[frame] = image_lib.rectify_spatial(order.ffObjImg[frame], polyVals2)

                if eta is not None:
                    if frame == 'B':
                        if config.params['onoff'] == True:
                            order.etaImgB     = order.etaImg
                            order.ffEtaImgB   = order.ffEtaImg
                        else:
                            order.etaImgB     = image_lib.rectify_spatial(order.etaImgB, polyVals1)
                            order.ffEtaImgB   = image_lib.rectify_spatial(order.ffEtaImgB, polyVals2)

                    else:
                        order.etaImg      = image_lib.rectify_spatial(order.etaImg, polyVals1)
                        order.ffEtaImg    = image_lib.rectify_spatial(order.ffEtaImg, polyVals2)
            else:
                order.objImg[frame]   = image_lib.rectify_spatial(order.objImg[frame], polyVals1)
                order.ffObjImg[frame] = image_lib.rectify_spatial(order.ffObjImg[frame], polyVals2)
        except:
            logger.warning('could not rectify using object trace, falling back to edge trace')
            order.objImg[frame]   = image_lib.rectify_spatial(order.objImg[frame], 
                                                              order.flatOrder.smoothedSpatialTrace)
            order.ffObjImg[frame] = image_lib.rectify_spatial(order.ffObjImg[frame], 
                                                              order.flatOrder.smoothedSpatialTrace)
            if eta is not None:
                    if frame == 'B':
                        order.etaImgB     = image_lib.rectify_spatial(order.etaImgB, 
                                                                      order.flatOrder.smoothedSpatialTrace)
                        order.ffEtaImgB   = image_lib.rectify_spatial(order.ffEtaImgB, 
                                                                      order.flatOrder.smoothedSpatialTrace)

                    else:
                        order.etaImg      = image_lib.rectify_spatial(order.etaImg, 
                                                                      order.flatOrder.smoothedSpatialTrace)
                        order.ffEtaImg    = image_lib.rectify_spatial(order.ffEtaImg, 
                                                                      order.flatOrder.smoothedSpatialTrace)


    order.spatialRectified = True
    logger.info('order has been rectified in the spatial dimension')
        
    return   
 

    
def __trim(order, eta=None):
    """
    """
    for frame in order.frames:
        if frame == 'AB': continue
        order.objImg[frame]   = order.objImg[frame][order.flatOrder.botTrim:order.flatOrder.topTrim, :]
        order.ffObjImg[frame] = order.ffObjImg[frame][order.flatOrder.botTrim:order.flatOrder.topTrim, :]

        if eta is not None:
            if frame == 'B':
                order.etaImgB   = order.etaImgB[order.flatOrder.botTrim:order.flatOrder.topTrim, :]
                order.ffEtaImgB = order.ffEtaImgB[order.flatOrder.botTrim:order.flatOrder.topTrim, :]
            else:
                order.etaImg    = order.etaImg[order.flatOrder.botTrim:order.flatOrder.topTrim, :]
                order.ffEtaImg  = order.ffEtaImg[order.flatOrder.botTrim:order.flatOrder.topTrim, :]
        
    return



def __rectify_spectral(order, eta=None):
    """
    """   
    for frame in order.frames:
        #print('FRAME', frame)
        if frame == 'AB': continue
        if config.params['onoff'] == True and frame == 'B':
            order.objImg[frame], peak1   = image_lib.rectify_spectral(order.objImg[frame], order.spectralTrace['A'], returnpeak=True)
            order.ffObjImg[frame], peak2 = image_lib.rectify_spectral(order.ffObjImg[frame], order.spectralTrace['A'], returnpeak=True)
        else:
            order.objImg[frame], peak1   = image_lib.rectify_spectral(order.objImg[frame], order.spectralTrace[frame], returnpeak=True)
            order.ffObjImg[frame], peak2 = image_lib.rectify_spectral(order.ffObjImg[frame], order.spectralTrace[frame], returnpeak=True)

        if frame == 'A':
            order.flatOrder.rectFlatImg, peak0 = image_lib.rectify_spectral(order.flatOrder.rectFlatImg, order.spectralTrace[frame], returnpeak=True)

        if eta is not None:
            if frame == 'B':
                if config.params['onoff'] == True:
                    order.etaImgB   = order.etaImg
                    order.ffEtaImgB = iorder.ffEtaImg
                else:
                    order.etaImgB   = image_lib.rectify_spectral(order.etaImgB, order.spectralTrace[frame], peak1)
                    order.ffEtaImgB = image_lib.rectify_spectral(order.ffEtaImgB, order.spectralTrace[frame], peak2)
            else:
                order.etaImg    = image_lib.rectify_spectral(order.etaImg, order.spectralTrace[frame], peak1)
                order.ffEtaImg  = image_lib.rectify_spectral(order.ffEtaImg, order.spectralTrace[frame], peak2)
    
    return     


              
def __extract_spectra(order, eta=None):
    
    if order.isPair:        
        # get object extraction range for AB
        order.objWindow['AB'], _, _ = \
                image_lib.get_extraction_ranges(order.objImg['AB'].shape[0], 
                order.peakLocation['AB'], config.params['obj_window'], None, None)   
                
        logger.info('frame AB extraction window width = {}'.format(str(len(order.objWindow['AB']))))

        # extract object spectrum from AB
        order.objSpec['AB'] = np.sum(order.ffObjImg['AB'][i, :] for i in order.objWindow['AB']) 
        
        frames = ['A', 'B']
    else:
        frames = ['A']
        
    for frame in frames:

        # get sky extraction ranges for A or A and B
        order.objWindow[frame], order.topSkyWindow[frame], order.botSkyWindow[frame] = \
                image_lib.get_extraction_ranges(order.objImg[frame].shape[0], 
                order.peakLocation[frame], config.params['obj_window'], 
                config.params['sky_window'], config.params['sky_separation'])
                
        logger.info('frame {} extraction window width = {}'.format(
                frame, str(len(order.objWindow[frame]))))
        logger.info('frame {} top background window width = {}'.format(
                frame, str(len(order.topSkyWindow[frame]))))
        if len(order.topSkyWindow[frame]) > 0:
            logger.info('frame {} top background window separation = {}'.format(
                    frame, str(order.topSkyWindow[frame][0] - order.objWindow[frame][-1])))
        logger.info('frame {} bottom background window width = {}'.format(
                frame, str(len(order.botSkyWindow[frame]))))
        if len(order.botSkyWindow[frame]) > 0:
            logger.info('frame {} bottom background window separation = {}'.format(
                    frame, str(order.objWindow[frame][0] - order.botSkyWindow[frame][-1])))       
                
        if eta is not None:
            if frame == 'B':
                # extract object, sky, etalon, and noise spectra for A and B and flat spectrum
                order.objSpec[frame], order.flatSpec, order.etalonSpec, order.skySpec[frame], order.noiseSpec[frame], \
                        order.topBgMean[frame], order.botBgMean[frame] = image_lib.extract_spectra(
                                order.ffObjImg[frame], order.flatOrder.rectFlatImg, order.noiseImg[frame], 
                                order.objWindow[frame], order.topSkyWindow[frame], 
                                order.botSkyWindow[frame], eta=order.ffEtaImgB) 
            else:
                # extract object, sky, etalon, and noise spectra for A and B and flat spectrum
                order.objSpec[frame], order.flatSpec, order.etalonSpec, order.skySpec[frame], order.noiseSpec[frame], \
                        order.topBgMean[frame], order.botBgMean[frame] = image_lib.extract_spectra(
                                order.ffObjImg[frame], order.flatOrder.rectFlatImg, order.noiseImg[frame], 
                                order.objWindow[frame], order.topSkyWindow[frame], 
                                order.botSkyWindow[frame], eta=order.ffEtaImg) 

        else:
            # extract object, sky, and noise spectra for A and B and flat spectrum
            order.objSpec[frame], order.flatSpec, order.skySpec[frame], order.noiseSpec[frame], \
                    order.topBgMean[frame], order.botBgMean[frame] = image_lib.extract_spectra(
                            order.ffObjImg[frame], order.flatOrder.rectFlatImg, order.noiseImg[frame], 
                            order.objWindow[frame], order.topSkyWindow[frame], 
                            order.botSkyWindow[frame])  

    ### XXX TESTING AREA
    '''
    # Try the defringe filter
    import matplotlib.pyplot as plt
    from scipy import fftpack
    from scipy import signal

    for frame in frames:
        if order.flatOrder.orderNum == 33:

            #fit1 = np.polyfit(np.arange(len(order.objSpec['A'])), order.objSpec['A'], 10)
            #z1   = np.poly1d(fit1)

            #orderspec2 = order.objSpec[frame]# - z1(np.arange(len(order.objSpec['A'])))

            #plt.plot(order.objSpec[frame], 'C0')
            #plt.plot(z1(np.arange(len(order.objSpec['A']))), 'C1')


            """
            f, Pxx_den = signal.periodogram(orderspec2, fs=len(order.objSpec[frame]))
            plt.figure(2)
            plt.semilogy(f, Pxx_den)
            #plt.ylim([1e-7, 1e2])
            plt.xlabel('frequency [Hz]')
            plt.ylabel('PSD [V**2/Hz]')
            """
            ## IDL CODE
            xdim   = len(order.objSpec[frame])
            nfil   = xdim/2 + 1
            cal1   = order.objSpec[frame]
            """
            f_high = 130
            f_low  = 150
            freq   = np.arange(nfil/2+1) / (nfil / float(xdim))
            fil    = np.zeros(len(freq))
            fil[np.where((freq > f_low) | (freq < f_high))] = 1
            fil    = np.append(fil, np.flip(fil[1:]))
            fil    = fftpack.ifft(fil)
            fil    = fil/nfil
            fil    = np.roll(fil, nfil/2)
            fil    = fil*np.hanning(nfil)
            """

            ##
            """
            cal1orig    = order.objSpec[frame]
            freq        = np.arange(nfil)
            cal1origmed = np.median(cal1orig)
            cal1smooth  = scipy.ndimage.median_filter(cal1, size=30)
            cal1fft     = fftpack.rfft(cal1orig-cal1smooth)
            yp          = abs(cal1fft[0:nfil])**2
            ypmax       = np.max(yp)
            yp          = yp / ypmax
            plt.figure(1011)
            plt.plot(freq, yp)
            """
            ##
            """
            cal1smooth = scipy.ndimage.median_filter(cal1, size=30)
            cal2smooth = scipy.ndimage.median_filter(cal2, size=30)

            cal1       = np.convolve(cal1-cal1smooth, fil, mode='same') + cal1smooth
            cal2       = np.convolve(cal2-cal2smooth, fil, mode='same') + cal2smooth
            #cal2smooth = median(cal2,30)
            #cal2       = convol(cal2-cal2smooth,fil,/edge_wrap)+cal2smooth
            plt.figure(101)
            plt.plot(cal1)
            plt.figure(102)
            plt.plot(cal2)
            plt.figure(103)
            plt.plot(cal1-cal2)
            #plt.show()
            #sys.exit()
            """
            ## IDL CODE
            """
            W        = fftpack.fftfreq(order.objSpec[frame].size, d=1./1024)
            toplot   = np.fft.fftshift(W)
            f_signal = fftpack.rfft(order.objSpec[frame])
            W1       = np.arange(len(W))
            
            plt.figure(3)
            plt.plot(W1, f_signal)
            
            plt.figure(4)
            plt.plot(toplot, f_signal)
            """
            f_high   = 300
            f_low    = 260
            W        = fftpack.fftfreq(order.objSpec[frame].size, d=1./1024)
            fftval   = fftpack.rfft(order.objSpec[frame])
            fftval[np.where((W > f_low) & (W < f_high))] = 0
            newsig   = fftpack.irfft(fftval)  
            """
            plt.figure(5)
            plt.plot(newsig)
            plt.show()
            sys.exit()
            """
            order.objSpec[frame] = newsig
    
    '''
    ### XXX TESTING AREA
        
    if order.isPair:
        #order.noiseSpec['AB'] = order.noiseSpec['A'] + order.noiseSpec['B']
        order.noiseSpec['AB'] = np.sqrt(np.square(order.noiseSpec['A']) + np.square(order.noiseSpec['B']))
        order.skySpec['AB']   = order.skySpec['A'] + order.skySpec['B'] # XXX This is a dirty fix
        
            
    return



def __calc_approximate_snr(order):
    
    if order.isPair:
        frames = ['A', 'B']
    else:
        frames = ['A']
        
    for frame in frames:
        
        bg = 0.0
        
        if order.topBgMean[frame] is not None:
            bg += order.topBgMean[frame]
        if order.botBgMean[frame] is not None:
            bg += order.botBgMean[frame]
        if order.topBgMean[frame] is not None and order.botBgMean[frame] is not None:
            bg /= 2
            
        order.snr[frame] = np.absolute(
                np.mean(order.ffObjImg[frame]\
                        [order.peakLocation[frame] : order.peakLocation[frame] + 1, :]) / bg)   
        
        logger.info('frame {} signal-to-noise ratio = {:.1f}'.format(frame, order.snr[frame]))
        
    if order.isPair:
        order.snr['AB'] = 0.0
        if order.snr['A'] is not None:
            order.snr['AB'] = order.snr['A']
        if order.snr['B'] is not None:
            order.snr['AB'] = order.snr['AB'] + order.snr['B']
        if order.snr['A'] is not None and order.snr['B'] is not None:
            order.snr['AB'] = order.snr['AB'] / 2.0
        
    return
    


def __characterize_spatial_profile(order):
    
    for frame in order.frames:
        try:
            if frame == 'B' and config.params['onoff']  == True:
                logger.debug('using frame A window width = {}'.format(abs(order.gaussianParams['A'][2])))
                order.gaussianParams[frame] = order.gaussianParams['A']
            else:
                for w in range(10, 30, 10):
                    logger.debug('gaussian window width = {}'.format(2 * w))
                    x0 = max(0, order.peakLocation[frame] - w)
                    x1 = min(len(order.spatialProfile[frame]) - 1, order.peakLocation[frame] + w)
                    x  = range(x1 - x0)
                    order.gaussianParams[frame], pcov = scipy.optimize.curve_fit(
                            image_lib.gaussian, x, order.spatialProfile[frame][x0:x1] - \
                            np.amin(order.spatialProfile[frame][x0:x1]))
                    order.gaussianParams[frame][1] += x0
                    if order.gaussianParams[frame][2] > 1.0:
                        break
        except Exception as e:
            logger.warning('cannot fit frame {} spatial profile to Gaussian'.format(frame))
            order.gaussianParams[frame] = None
        else:
            logger.info('frame {} spatial peak width = {:.1f} pixels'.format(
                    frame, abs(order.gaussianParams[frame][2])))
        
    return



def __find_spatial_profile_and_peak(order):
    """
    """
    
    MARGIN = 5
    
    for frame in order.frames:
        
        # find spatial profile(s)
        if frame == 'B' and config.params['onoff'] == True:
            order.spatialProfile[frame] = order.spatialProfile['A']
            order.peakLocation[frame]   = order.peakLocation['A']
            order.centroid[frame]       = order.centroid['A']

        else:
            order.spatialProfile[frame] = order.ffObjImg[frame].mean(axis=1)
            if len(order.spatialProfile[frame]) < (2 * MARGIN) + 2:
                raise DrpException.DrpException(
                        'cannot find spatial profile for frame {} order {}'.format(
                        frame, order.flatOrder.orderNum))
                
            # find peak locations
            order.peakLocation[frame] = np.argmax(order.spatialProfile[frame][MARGIN:-MARGIN]) + MARGIN
            logger.info('frame {} spatial profile peak intensity row {:d}'.format(
                    frame, order.peakLocation[frame]))
        
            # fit peak to Gaussian, save Gaussian parameters and real centroid location
            p0 = order.peakLocation[frame] - (config.params['obj_window'] // 2)
            p1 = order.peakLocation[frame] + (config.params['obj_window'] // 2)
            order.centroid[frame] = (scipy.ndimage.measurements.center_of_mass(
                order.spatialProfile[frame][p0:p1]))[0] + p0 
            logger.info('frame {} spatial profile peak centroid row {:.1f}'.format(
                    frame, float(order.centroid[frame])))     
    
    return



def __calculate_SNR(order):
    
    bg = 0.0

    if order.topBgMean is not None:
        bg += order.topBgMean
    if order.botBgMean is not None:
        bg += order.botBgMean
    if order.topBgMean is not None and order.botBgMean is not None:
        bg /= 2
    order.snr = np.absolute(np.mean(
            order.ffObjImg['A'][order.peakLocation['A']:order.peakLocation['A'] + 1, :]) / bg)
    
    
    