import logging
import numpy as np
import scipy.optimize as op
import image_lib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


def NormDist(x, mean, sigma, baseline, amplitude):

	Dist1 = amplitude * 1. / np.sqrt(2. * np.pi * sigma**2) * np.exp(-1*(x - mean)**2 / (2.*sigma**2) ) + baseline

	return Dist1



def CreateSpatialMap(order):

	for frame in order.frames:

		# I think this is the image?
		image   = order.objImg[frame]
		ffimage = order.ffObjImg[frame]

		#print(image.shape)
		Pixels = []
		for i in range(ffimage.shape[1]):
			#print(i)
			guess1 = np.where(ffimage[:,i] == np.max(ffimage[:,i]))[0][0]
			#print(guess1)
			try:
				popt, pcov = op.curve_fit(NormDist, np.arange(len(ffimage[:,i])), ffimage[:,i], p0=[guess1, 2, np.median(ffimage[:,i]), np.max(ffimage[:,i])], maxfev=10000) # Where should a pixel start? (0, 1, 0.5?)
				#print(popt)
				#print(pcov)
				Pixels.append(popt[0])
			except: 
				Pixels.append(0)
			#if i > 270:
		#		plt.plot(image[:,i])
	#			plt.plot(np.arange(len(image[:,i])), NormDist(np.arange(len(image[:,i])), *popt), 'r-', lw=0.5)
	#			plt.show()
			#sys.exit()

		Pixels = np.array(Pixels)
		pixels = np.arange(ffimage.shape[1])

		unFitted = True
		count = 0

		#fig = plt.figure(3)
		#ax = fig.add_subplot(111)
		#ax.scatter(pixels, Pixels, c='0.5', s=3, alpha=0.5)

		while unFitted:
			#plt.figure(3)
			#plt.scatter(pixels, Pixels, c='0.5')
			z1 = np.polyfit(pixels, Pixels, 3)
			p1 = np.poly1d(z1)
			#plt.plot(pixels, p1(pixels), 'r--')
			#plt.show()
			#sys.exit()

			# Do the fit
			hist = Pixels - p1(pixels)
			sig  = np.std(hist)
			if count == 0: 
				ind1 = np.where(abs(hist) < 2*sig)
			else:
				ind1 = np.where(abs(hist) < 5*sig)

			#plt.scatter(pixels[ind1], Pixels[ind1], marker='x', c='r')
			#plt.show()
			newpix = pixels[ind1].flatten()
			newPix = Pixels[ind1].flatten()

			#print(len(Pixels) == len(newPix))
			#print(unFitted)
			if len(Pixels) == len(newPix): 
				#ax.plot(np.arange(ffimage.shape[1]), p1(np.arange(ffimage.shape[1])), 'r--')
				#ax.scatter(newpix, newPix, marker='x', c='b', s=3)

				# Calc the RMS
				rms   = np.sqrt(np.mean(hist**2))
				sumsq = np.sum(hist**2)
				#print(rms, sumsq)
				#print(z1)

				#at = AnchoredText('RMS = %0.4f'%(rms) + '\n' + 'Coeff: %0.3E %0.3E %0.3E %0.3E'%(z1[-1], z1[-2], z1[-3], z1[-4]),
				#    prop=dict(size=8), frameon=False,
				#    loc=2,
				#    )
				"""
				at = AnchoredText('Sum of Squared Errors =  %0.4f'%(sumsq) + \
				              '\n' + 'Coeff =' + \
				              '\n' + '%0.3E'%(z1[-1]) + \
				              '\n' + '%0.3E'%(z1[-2]) + \
				              '\n' + '%0.3E'%(z1[-3]) + \
				              '\n' + '%0.3E'%(z1[-4]),
				prop=dict(size=8), frameon=False,
				loc=2,
				)

				ax.add_artist(at)
				"""
				unFitted = False


			#print(unFitted)
			pixels = pixels[ind1].flatten()
			Pixels = Pixels[ind1].flatten()

			count+=1
		"""
		plt.figure(1)
		plt.imshow(image, origin='lower')

		plt.figure(2)
		plt.imshow(ffimage, origin='lower')

		plt.show()
        """
		return p1(np.arange(ffimage.shape[1]))

        


def spectral_trace(calimage, linelist='apohenear.dat', interac=True,
                   fmask=(1,), display=True,
                   tol=10, fit_order=2, previous='', mode='poly',
                   second_pass=True):
    """
    Determine the wavelength solution to be used for the science images.
    Can be done either automatically (buyer beware) or manually. Both the
    manual and auto modes use a "slice" through the chip center to learn
    the wavelengths of specific lines. Emulates the IDENTIFY
    function in IRAF.

    If the automatic mode is selected (interac=False), program tries to
    first find significant peaks in the "slice", then uses a brute-force
    guess scheme based on the grating information in the header. While
    easy, your mileage may vary with this method.

    If the interactive mode is selected (interac=True), you click on
    features in the "slice" and identify their wavelengths.

    Parameters
    ----------
    calimage : str
        Etalon lamp image
    linelist : str, optional
        The linelist file to use in the resources/linelists/ directory.
        Only used in automatic mode. (Default is etalon.dat)
    interac : bool, optional
        Should the etalon identification be done interactively (manually)?
        (Default is True)
    trim : bool, optional
        Trim the image using the DATASEC keyword in the header, assuming
        has format of [0:1024,0:512] (Default is False) XXX remove this option, trim already done
    fmask : array-like, optional
        A list of illuminated rows in the spatial direction (Y), as
        returned by flatcombine.
    display : bool, optional
    tol : int, optional
        When in automatic mode, the tolerance in pixel units between
        linelist entries and estimated wavelengths for the first few
        lines matched... use carefully. (Default is 10)
    mode : str, optional
        What type of function to use to fit the entire 2D wavelength
        solution? Options include (poly, spline2d). (Default is poly)
    fit_order : int, optional
        The polynomial order to use to interpolate between identified
        peaks in the HeNeAr (Default is 2)
    previous : string, optional
        name of file containing previously identified peaks. Still has to
        do the fitting.

    Returns
    -------
    wfit : bivariate spline object or 2d polynomial
        The wavelength solution at every pixel. Output type depends on the
        mode keyword above (poly is recommended)
    """

    print('Finding etalon lines')

    # silence the polyfit warnings
    warnings.simplefilter('ignore', np.RankWarning)

    img = calimage

    # this approach will be very DIS specific
    #disp_approx = hdu[0].header['DISPDW']
    #wcen_approx = hdu[0].header['DISPWC']
    disp_approx = 1.66
    wcen_approx = 7300

    # take a slice thru the data (+/- 10 pixels) in center row of chip
    slice = img[img.shape[0]/2-10:img.shape[0]/2+10,:].sum(axis=0)

    # use the header info to do rough solution (linear guess)
    wtemp = (np.arange(len(slice))-len(slice)/2) * disp_approx * sign + wcen_approx


    ######   IDENTIFY   (auto and interac modes)
    # = = = = = = = = = = = = = = = =
    #-- automatic mode
    if (interac is False) and (len(previous)==0):
        print("Doing automatic wavelength calibration on HeNeAr.")
        print("Note, this is not very robust. Suggest you re-run with interac=True")
        # find the linelist of choice

        linelists_dir = os.path.dirname(os.path.realpath(__file__))+ '/resources/linelists/'
        # if (len(linelist)==0):
        #     linelist = os.path.join(linelists_dir, linelist)

        # import the linelist
        linewave = np.loadtxt(os.path.join(linelists_dir, linelist), dtype='float',
                              skiprows=1,usecols=(0,),unpack=True)


        pcent_pix, wcent_pix = find_peaks(wtemp, slice, pwidth=10, pthreshold=97)

    #   loop thru each peak, from center outwards. a greedy solution
    #   find nearest list line. if not line within tolerance, then skip peak
        pcent = []
        wcent = []

        # find center-most lines, sort by dist from center pixels
        ss = np.argsort(np.abs(wcent_pix-wcen_approx))

        #coeff = [0.0, 0.0, disp_approx*sign, wcen_approx]
        coeff = np.append(np.zeros(fit_order-1),(disp_approx*sign, wcen_approx))

        for i in range(len(pcent_pix)):
            xx = pcent_pix-len(slice)/2
            #wcent_pix = coeff[3] + xx * coeff[2] + coeff[1] * (xx*xx) + coeff[0] * (xx*xx*xx)
            wcent_pix = np.polyval(coeff, xx)

            if display is True:
                plt.figure()
                plt.plot(wtemp, slice, 'b')
                plt.scatter(linewave,np.ones_like(linewave)*np.nanmax(slice),marker='o',c='cyan')
                plt.scatter(wcent_pix,np.ones_like(wcent_pix)*np.nanmax(slice)/2.,marker='*',c='green')
                plt.scatter(wcent_pix[ss[i]],np.nanmax(slice)/2., marker='o',c='orange')

            # if there is a match w/i the linear tolerance
            if (min((np.abs(wcent_pix[ss][i] - linewave))) < tol):
                # add corresponding pixel and *actual* wavelength to output vectors
                pcent = np.append(pcent,pcent_pix[ss[i]])
                wcent = np.append(wcent, linewave[np.argmin(np.abs(wcent_pix[ss[i]] - linewave))] )

                if display is True:
                    plt.scatter(wcent,np.ones_like(wcent)*np.nanmax(slice),marker='o',c='red')

                if (len(pcent)>fit_order):
                    coeff = np.polyfit(pcent-len(slice)/2, wcent, fit_order)

            if display is True:
                plt.xlim((min(wtemp),max(wtemp)))
                plt.show()

        print('Matches')
        print(pcent)
        print(wcent)
        lout = open(calimage+'.lines', 'w')
        lout.write("# This file contains the HeNeAr lines identified [auto] Columns: (pixel, wavelength) \n")
        for l in range(len(pcent)):
            lout.write(str(pcent[l]) + ', ' + str(wcent[l])+'\n')
        lout.close()

        # the end result is the vector "coeff" has the wavelength solution for "slice"
        # update the "wtemp" vector that goes with "slice" (fluxes)
        wtemp = np.polyval(coeff, (np.arange(len(slice))-len(slice)/2))


    # = = = = = = = = = = = = = = = =
    #-- manual (interactive) mode
    elif (interac is True):
        if (len(previous)==0):
            print('')
            print('Using INTERACTIVE HeNeAr_fit mode:')
            print('1) Click on HeNeAr lines in plot window')
            print('2) Enter corresponding wavelength in terminal and press <return>')
            print('   If mis-click or unsure, just press leave blank and press <return>')
            print('3) To delete an entry, click on label, enter "d" in terminal, press <return>')
            print('4) Close plot window when finished')

            xraw = np.arange(len(slice))
            class InteracWave(object):
                # http://stackoverflow.com/questions/21688420/callbacks-for-graphical-mouse-input-how-to-refresh-graphics-how-to-tell-matpl
                def __init__(self):
                    self.fig = plt.figure()
                    self.ax = self.fig.add_subplot(111)
                    self.ax.plot(wtemp, slice, 'b')
                    plt.xlabel('Wavelength')
                    plt.ylabel('Counts')

                    self.pcent = [] # the pixel centers of the identified lines
                    self.wcent = [] # the labeled wavelengths of the lines
                    self.ixlib = [] # library of click points

                    self.cursor = Cursor(self.ax, useblit=False, horizOn=False,
                                         color='red', linewidth=1 )
                    self.connect = self.fig.canvas.mpl_connect
                    self.disconnect = self.fig.canvas.mpl_disconnect
                    self.clickCid = self.connect("button_press_event",self.OnClick)

                def OnClick(self, event):
                    # only do stuff if toolbar not being used
                    # NOTE: this subject to change API, so if breaks, this probably why
                    # http://stackoverflow.com/questions/20711148/ignore-matplotlib-cursor-widget-when-toolbar-widget-selected
                    if self.fig.canvas.manager.toolbar._active is None:
                        ix = event.xdata
                        print('onclick point:', ix)

                        # if the click is in good space, proceed
                        if (ix is not None) and (ix > np.nanmin(wtemp)) and (ix < np.nanmax(wtemp)):
                            # disable button event connection
                            self.disconnect(self.clickCid)

                            # disconnect cursor, and remove from plot
                            self.cursor.disconnect_events()
                            self.cursor._update()

                            # get points nearby to the click
                            nearby = np.where((wtemp > ix-10*disp_approx) &
                                              (wtemp < ix+10*disp_approx) )

                            # find if click is too close to an existing click (overlap)
                            kill = None
                            if len(self.pcent)>0:
                                for k in range(len(self.pcent)):
                                    if np.abs(self.ixlib[k]-ix)<tol:
                                        kill_d = raw_input('> WARNING: Click too close to existing point. To delete existing point, enter "d"')
                                        print('You entered:', kill_d, kill_d=='d')
                                        if kill_d=='d':
                                            kill = k
                                if kill is not None:
                                    del(self.pcent[kill])
                                    del(self.wcent[kill])
                                    del(self.ixlib[kill])


                            # If there are enough valid points to possibly fit a peak too...
                            if (len(nearby[0]) > 4) and (kill is None):
                                print('Fitting Peak')
                                imax = np.nanargmax(slice[nearby])

                                pguess = (np.nanmax(slice[nearby]), np.median(slice), xraw[nearby][imax], 2.)
                                try:
                                    popt,pcov = curve_fit(_gaus, xraw[nearby], slice[nearby], p0=pguess)
                                    self.ax.plot(wtemp[int(popt[2])], popt[0], 'r|')
                                except ValueError:
                                    print('> WARNING: Bad data near this click, cannot centroid line with Gaussian. I suggest you skip this one')
                                    popt = pguess
                                except RuntimeError:
                                    print('> WARNING: Gaussian centroid on line could not converge. I suggest you skip this one')
                                    popt = pguess

                                # using raw_input sucks b/c doesn't raise terminal, but works for now
                                try:
                                    number=float(raw_input('> Enter Wavelength: '))
                                    print('Pixel Value:',popt[2])
                                    self.pcent.append(popt[2])
                                    self.wcent.append(number)
                                    self.ixlib.append((ix))
                                    self.ax.plot(wtemp[int(popt[2])], popt[0], 'ro')
                                    print('  Saving '+str(number))
                                except ValueError:
                                    print("> Warning: Not a valid wavelength float!")

                            elif (kill is None):
                                print('> Error: No valid data near click!')

                            # reconnect to cursor and button event
                            self.clickCid = self.connect("button_press_event",self.OnClick)
                            self.cursor = Cursor(self.ax, useblit=False,horizOn=False,
                                             color='red', linewidth=1 )
                    else:
                        pass

            # run the interactive program
            wavefit = InteracWave()
            plt.show() #activate the display - GO!

            # how I would LIKE to do this interactively:
            # inside the interac mode, do a split panel, live-updated with
            # the wavelength solution, and where user can edit the fit_order

            # how I WILL do it instead
            # a crude while loop here, just to get things moving

            # after interactive fitting done, get results fit peaks
            pcent = np.array(wavefit.pcent,dtype='float')
            wcent = np.array(wavefit.wcent, dtype='float')

            print('> You have identified '+str(len(pcent))+' lines')
            lout = open(calimage+'.lines', 'w')
            lout.write("# This file contains the HeNeAr lines identified [manual] Columns: (pixel, wavelength) \n")
            for l in range(len(pcent)):
                lout.write(str(pcent[l]) + ', ' + str(wcent[l])+'\n')
            lout.close()


        if (len(previous)>0):
            pcent, wcent = np.loadtxt(previous, dtype='float',
                                      unpack=True, skiprows=1,delimiter=',')


        #---  FIT SMOOTH FUNCTION ---

        # fit polynomial thru the peak wavelengths
        # xpix = (np.arange(len(slice))-len(slice)/2)
        # coeff = np.polyfit(pcent-len(slice)/2, wcent, fit_order)
        xpix = np.arange(len(slice))
        coeff = np.polyfit(pcent, wcent, fit_order)
        wtemp = np.polyval(coeff, xpix)

        done = str(fit_order)
        while (done != 'd'):
            fit_order = int(done)
            coeff = np.polyfit(pcent, wcent, fit_order)
            wtemp = np.polyval(coeff, xpix)

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.plot(pcent, wcent, 'bo')
            ax1.plot(xpix, wtemp, 'r')

            ax2.plot(pcent, wcent - np.polyval(coeff, pcent),'ro')
            ax2.set_xlabel('pixel')
            ax1.set_ylabel('wavelength')
            ax2.set_ylabel('residual')
            ax1.set_title('fit_order = '+str(fit_order))

            # ylabel('wavelength')

            print(" ")
            print('> How does this look?  Enter "d" to be done (accept), ')
            print('  or a number to change the polynomial order and re-fit')
            print('> Currently fit_order = '+str(fit_order))
            print(" ")

            plt.show(block=False)

            _CheckMono(wtemp)

            print(' ')
            done = str(raw_input('ENTER: "d" (done) or a # (poly order): '))


    # = = = = = = = = = = = = = = = = = =
    # now rough wavelength is found, either via interactive or auto mode!

    #-- SECOND PASS
    second_pass = False
    if second_pass is True:
        linelists_dir = os.path.dirname(os.path.realpath(__file__))+ '/resources/linelists/'
        hireslinelist = 'henear.dat'
        linewave2 = np.loadtxt(os.path.join(linelists_dir, hireslinelist), dtype='float',
                               skiprows=1, usecols=(0,), unpack=True)

        tol2 = tol # / 2.0
        print(wtemp)
        """
        plt.figure(999)
        plt.plot(wtemp, slice)
        plt.show()
        """
        pcent_pix2, wcent_pix2 = find_peaks(wtemp, slice, pwidth=10, pthreshold=80)
        print(pcent_pix2)
        print(wcent_pix2)

        pcent2 = []
        wcent2 = []
        # sort from center wavelength out
        ss = np.argsort(np.abs(wcent_pix2-wcen_approx))

        # coeff should already be set by manual or interac mode above
        # coeff = np.append(np.zeros(fit_order-1),(disp_approx*sign, wcen_approx))
        for i in range(len(pcent_pix2)):
            xx = pcent_pix2-len(slice)/2
            wcent_pix2 = np.polyval(coeff, xx)

            if (min((np.abs(wcent_pix2[ss][i] - linewave2))) < tol2):
                # add corresponding pixel and *actual* wavelength to output vectors
                pcent2 = np.append(pcent2, pcent_pix2[ss[i]])
                wcent2 = np.append(wcent2, linewave2[np.argmin(np.abs(wcent_pix2[ss[i]] - linewave2))] )
                #print(pcent2, wcent2)

            #-- update in real time. maybe not good for 2nd pass
            # if (len(pcent2)>fit_order):
            #     coeff = np.polyfit(pcent2-len(slice)/2, wcent2, fit_order)

            if display is True:
                plt.figure()
                plt.plot(wtemp, slice, 'b')
                plt.scatter(linewave2,np.ones_like(linewave2)*np.nanmax(slice),
                            marker='o',c='cyan')
                plt.scatter(wcent_pix2,np.ones_like(wcent_pix2)*np.nanmax(slice)/2.,
                            marker='*',c='green')
                plt.scatter(wcent_pix2[ss[i]],np.nanmax(slice)/2.,
                            marker='o',c='orange')
                plt.text(np.nanmin(wcent_pix2), np.nanmax(slice)*0.95, hireslinelist)
                plt.text(np.nanmin(wcent_pix2), np.nanmax(slice)/2.*1.1, 'detected lines')

                plt.scatter(wcent2,np.ones_like(wcent2)*np.nanmax(slice)*1.05,marker='o',c='red')
                plt.text(np.nanmin(wcent_pix2), np.nanmax(slice)*1.1, 'matched lines')

                plt.ylim((np.nanmin(slice), np.nanmax(slice)*1.2))
                plt.xlim((min(wtemp),max(wtemp)))
                plt.show()
        wtemp = np.polyval(coeff, (np.arange(len(slice))-len(slice)/2))

        lout = open(calimage+'.lines2', 'w')
        lout.write("# This file contains the HeNeAr lines identified [2nd pass] Columns: (pixel, wavelength) \n")
        for l in range(len(pcent2)):
            lout.write(str(pcent2[l]) + ', ' + str(wcent2[l])+'\n')
        lout.close()

        xpix = np.arange(len(slice))
        coeff = np.polyfit(pcent2, wcent2, fit_order)
        wtemp = np.polyval(coeff, xpix)


        #---  FIT SMOOTH FUNCTION ---
        if interac is True:
            done = str(fit_order)
            while (done != 'd'):
                fit_order = int(done)
                coeff = np.polyfit(pcent2, wcent2, fit_order)
                wtemp = np.polyval(coeff, xpix)

                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                ax1.plot(pcent2, wcent2, 'bo')
                ax1.plot(xpix, wtemp, 'r')

                ax2.plot(pcent2, wcent2 - np.polyval(coeff, pcent2),'ro')
                ax2.set_xlabel('pixel')
                ax1.set_ylabel('wavelength')
                ax2.set_ylabel('residual')
                ax1.set_title('2nd pass, fit_order = '+str(fit_order))

                # ylabel('wavelength')

                print(" ")
                print('> How does this look?  Enter "d" to be done (accept), ')
                print('  or a number to change the polynomial order and re-fit')
                print('> Currently fit_order = '+str(fit_order))
                print(" ")

                plt.show(block=False)

                _CheckMono(wtemp)

                print(' ')
                done = str(raw_input('ENTER: "d" (done) or a # (poly order): '))

    #-- trace the peaks vertically --
    xcent_big, ycent_big, wcent_big = line_trace(img, pcent, wcent,
                                                 fmask=fmask, display=display)

    #-- turn these vertical traces in to a whole chip wavelength solution
    wfit = lines_to_surface(img, xcent_big, ycent_big, wcent_big,
                            mode=mode, fit_order=fit_order)

    print('This is the FIT:', wfit)
    print(wfit.shape)
    plt.figure(1)
    plt.plot(wfit[0])
    plt.show()
    return wfit