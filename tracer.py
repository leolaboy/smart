import numpy as np
import scipy.ndimage

import CAT_Functions as cat
import scipy.optimize as op

def trace_edge(data, start, searchWidth, bgWidth, jumpThresh, eta=None):

    # initialize trace array
    trace = np.zeros(data.shape[1])

    # nJumps is the number of times successive centroids differed by more than threshold
    nJumps = 0

    # first centroid assumed to be at start
    trace[0] = start
        
    # find centroids for the rest of the columns in data

    if start == 49: 
        PLOT = False
        import matplotlib.pyplot as plt
        plt.imshow(data, origin='lower')
        plt.show(block=False)
    else: 
        PLOT = False

    if eta is not None: 
        stepcount = 5
    else: 
        stepcount = 1

    for i in range(1, data.shape[1]):
        #print('I',i)
        
        # define search window
        ymin = int(trace[i - 1] - searchWidth)
        ymax = int(trace[i - 1] + searchWidth)
        
        # clip search window at top and bottom of column
        if abs(ymax) > data.shape[0]:
            ymax = int(data.shape[0])

        if ymin < 1:  # don't let it trace the bottom of the detector
            ymin = 1
        if ymax <= 0:
            # how can this happen?
            ymax = int(trace[i] + searchWidth) + 1

        if bgWidth <= 0: 
            bgMean = 0.0;
        
        else:
            # If bgWidth > 0 then we will subtract average of pixel values at two locations
            # from each pixel value in search window.  Two locations are previous centroid
            # plus and minus search width.
        
            bgMin = trace[i - 1] - bgWidth
            if bgMin < 0:
                bgMin = 0
            bgMax = trace[i - 1] + bgWidth
            if bgMax > data.shape[0]:
                bgMax = data.shape[0] - 1
            
            try:
                bgMean = (data[bgMin, i] + data[bgMax, i]) / 2.0
            except:
                bgMean = 0.0


        if eta is not None:
            #print('TEST', data[int(ymin):int(ymax) + 1, i:i+stepcount] - bgMean)
            #print('TEST', np.sum(data[int(ymin):int(ymax) + 1, i:i+stepcount]))

            #print('TEST', np.sum(data[int(ymin):int(ymax) + 1, i:i+stepcount], axis=0))
            #print('TEST', np.sum(data[int(ymin):int(ymax) + 1, i:i+stepcount], axis=1))
            #sys.exit()
            #print(scipy.ndimage.measurements.center_of_mass(np.sum(data[int(ymin):int(ymax) + 1, i:i+stepcount], axis=1) - bgMean))
            #trace[i] = scipy.ndimage.measurements.center_of_mass(
            #              np.sum(data[int(ymin):int(ymax) + 1, i:i+stepcount], axis=1) - bgMean)[0] + ymin
            Xs = np.arange(len(np.sum(data[int(ymin):int(ymax) + 1, i:i+stepcount], axis=1)))
            Ys = np.sum(data[int(ymin):int(ymax) + 1, i:i+stepcount], axis=1) 
            guess1 = np.where(Ys == np.max(Ys))[0][0]
            popt, pcov = op.curve_fit(cat.NormDist, Xs, 
                                                    Ys, 
                                                    p0=[guess1, 2, np.median(Ys), np.max(Ys)], maxfev=1000000) # Where should a pixel start? (0, 1, 0.5?)
            trace[i] = popt[0] + ymin
        else: 
            trace[i] = scipy.ndimage.measurements.center_of_mass(
                              data[int(ymin):int(ymax) + 1, i] - bgMean)[0] + ymin
        
        # if PLOT:        
        #     import pylab as pl
        #     x0 = max(0, int(trace[i - 1]) - 50)
        #     x1 = min(1023, int(trace[i-1]) + 50)
        #     print(np.sum(data[int(ymin):int(ymax) + 1, i:i+stepcount], axis=1))
        #     print(np.sum(data[int(ymin):int(ymax) + 1, i:i+stepcount], axis=1))
        #     Xs = np.arange(len(np.sum(data[int(ymin):int(ymax) + 1, i:i+stepcount], axis=1)))
        #     Ys = np.sum(data[int(ymin):int(ymax) + 1, i:i+stepcount], axis=1) 
        #     print(Xs)
        #     print(Ys)
        #     #plt.figure(199)
        #     #plt.plot(Xs, Ys)
        #     #plt.show()
        #     #sys.exit()

        #     guess1 = np.where(Ys == np.max(Ys))[0][0]
        #     popt, pcov = op.curve_fit(cat.NormDist, Xs, 
        #                                             Ys, 
        #                                             p0=[guess1, 2, np.median(Ys), np.max(Ys)], maxfev=10000) # Where should a pixel start? (0, 1, 0.5?)
        #     print('1', popt)
        #     print('2', pcov)

        #     plt.figure(198)
        #     plt.plot(Xs, Ys)
        #     plt.plot(Xs, cat.NormDist(Xs, *popt), 'r--')
        #     plt.show()
        #     #sys.exit()    

            # pl.figure(101)
            # pl.cla()
            # pl.plot(np.arange(x0, x1), np.sum(data[x0:x1, i:i+5], axis=1))
            # pl.plot([trace[i], trace[i]], pl.ylim())
            # pl.plot([trace[i-1], trace[i-1]], pl.ylim(), 'r--')
            # pl.plot([ymin, ymin], pl.ylim(), 'r-')
            # pl.plot([ymax, ymax], pl.ylim(), 'r-')
            # #pl.plot([trace[i], trace[i]], [0, pl.ylim()[0]], 'g-')

            # pl.figure(102)
            # pl.plot(data[x0:x1, i], 'ro')
            # pl.plot(data[int(ymin):int(ymax) + 1, i] - bgMean, 'go')
            # pl.plot([trace[i], trace[i]], [0, pl.ylim()[0]], 'g-')
            # print(trace[max(0, i-10):i])
            # print(np.abs(trace[i] - trace[i - 1]), trace[i], trace[i-1], jumpThresh)
            # print('TEST2', np.flatnonzero(data[int(ymin):int(ymax) + 1, i] - bgMean).mean()+0.5)
            # print(nJumps)
            
            # pl.show()

        if trace[i] is np.inf or trace[i] is -np.inf:  
            # went off array
            print('went off array')
            return None, None
    
        # centroid jumped more than traceDelta
        if np.abs(trace[i] - trace[i - 1]) > jumpThresh:
            nJumps += 1
            if i > 4:
                # jump is past beginning, use past three centroids
                trace[i] = trace[i - 3:i - 1].mean()
            elif i > 1:
                # average as many traces as we have gone through
                trace[i] = trace[i - 2:i - 1].mean()
            else:
                # use the first one found
                trace[i] = trace[i - 1]

        if PLOT:
            #print('I2', i+np.floor(stepcount/2.))
            #plt.scatter(i+np.floor(stepcount/2.), trace[i], alpha=0.5)
            plt.scatter(i, trace[i], alpha=0.5)
        
    if PLOT:
        plt.show()
        sys.exit()

    return trace, nJumps

