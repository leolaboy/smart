import logging
import numpy as np
import image_lib


#logger = logging.getLogger('obj')

class FlatOrder:
    """
    
    Top refers to high row numbers, longer wavelengths.
    Bottom refers to low row numbers, shorter wavelengths.
    LHS refers to left hand side of order, low column numbers, shorter wavelengths.
    """
    
    def __init__(self, baseName, orderNum, logger):
        
        self.flatBaseName       = baseName
        self.orderNum           = orderNum
        self.logger             = logger
        
        self.valid              = False
        
        self.topCalc            = None  # LHS top row of order, according to grating eq
        self.botCalc            = None  # LHS bottom row of order, according to grating eq
        self.gratingEqWaveScale = None  # wavelength scale, according to grating eq
        
        self.topMeas            = None  # measured LHS top row of order
        self.botMeas            = None  # measured LHS bottom row of order
        
        self.topEdgeTrace = None         # top edge trace
        self.botEdgeTrace = None         # bot edge trace
        self.avgEdgeTrace = None

        self.longSlitEdgeMargin = 0
        self.cutoutPadding      = 0
        
        self.highestPoint = None
        self.lowestPoint  = None
        self.topTrim      = None
        self.botTrim      = None
                
        self.onOrderMask  = None
        self.offOrderMask = None
        
        self.mean   = None
        self.median = None
        
        self.cutout      = None
#         self.flatImg = None
        self.normFlatImg = None
        self.rectFlatImg = None
        
        self.normalized        = False
        self.spatialRectified  = False
        self.spectralRectified = False

        self.smoothedSpatialTrace    = None
        self.spatialTraceMask        = None
        self.spatialTraceFitResidual = None

        # set up for AB frames
        self.smoothedSpatialTraceA    = None
        self.spatialTraceMaskA        = None
        self.spatialTraceFitResidualA = None

        self.smoothedSpatialTraceB    = None
        self.spatialTraceMaskB        = None
        self.spatialTraceFitResidualB = None
        
        
    def reduce(self):
        
        self.logger.info('reducing flat order {}'.format(self.orderNum))
        
        # normalize flat
        self.normFlatImg, self.median =  image_lib.normalize(
                self.cutout, self.onOrderMask, self.offOrderMask)
        self.normalized = True
        self.logger.info('flat normalized, flat median = ' + str(round(self.median, 1)))
        
        # spatially rectify flat
        self.rectFlatImg  = image_lib.rectify_spatial(self.normFlatImg, self.smoothedSpatialTrace)
        self.rectFlatImgA = image_lib.rectify_spatial(self.normFlatImg, self.smoothedSpatialTraceA)
        self.rectFlatImgB = image_lib.rectify_spatial(self.normFlatImg, self.smoothedSpatialTraceB)

        self.spatialRectified = True
        
        # compute top and bottom trim points
        self.calcTrimPoints()
        self.calcTrimPointsAB()

        # trim rectified flat order images
        self.rectFlatImg  = self.rectFlatImg[self.botTrim:self.topTrim, :]
        self.rectFlatImgA = self.rectFlatImgA[self.botTrimA:self.topTrimA, :]
        self.rectFlatImgB = self.rectFlatImgB[self.botTrimB:self.topTrimB, :]
        
        self.logger.debug('reduction of flat order {} complete'.format(self.orderNum))
        
        return
    
    def calcTrimPoints(self):
        if self.lowestPoint > self.cutoutPadding:
            self.topTrim = self.highestPoint - self.lowestPoint + self.cutoutPadding - 3
        else:
            self.topTrim = self.highestPoint - 3
        h = np.amin(self.topEdgeTrace - self.botEdgeTrace)
        self.botTrim = self.topTrim - h + 3
        self.botTrim = int(max(0, self.botTrim))
        self.topTrim = int(min(self.topTrim, 1023))
        
        return

    def calcTrimPointsAB(self):
        if self.lowestPoint > self.cutoutPadding:
            self.topTrimA = self.highestPoint - self.lowestPoint + self.cutoutPadding - 3
        else:
            self.topTrimA = self.highestPoint - 3
        h = np.amin(self.topEdgeTrace - self.botEdgeTrace)
        self.botTrimA = self.topTrimA - h + 3
        self.botTrimA = int(max(0, self.botTrimA))
        self.topTrimA = int(min(self.topTrimA, 1023))

        if self.lowestPoint > self.cutoutPadding:
            self.topTrimB = self.highestPoint - self.lowestPoint + self.cutoutPadding - 3
        else:
            self.topTrimB = self.highestPoint - 3
        h = np.amin(self.topEdgeTrace - self.botEdgeTrace)
        self.botTrimB = self.topTrimB - h + 3
        self.botTrimB = int(max(0, self.botTrimA))
        self.topTrimB = int(min(self.topTrimB, 1023))
        
        return
        
        
        
        