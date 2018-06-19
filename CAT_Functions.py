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

