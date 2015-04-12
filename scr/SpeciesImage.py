from Components import *
from Curvature import Curvature
from Venation import Venation
from Morphology import *
#from Log import Log
import Constants as C
from cv2 import EM

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

class SpeciesImage(object):

    def __init__(self, path="", inputImg=None):
        self.lol = 0
        self.lol2 = 0
        self.lol3 = 0
        self.colorByCluster = {}
        self.distanceByCluster = {}
        self.freqByCluster = {}
        self.injuryByCluster = {}
        self.path = path
        if inputImg==None:
            self.originalSize = loadImage(path)
        else:
            self.originalSize = inputImg
        self.original = resizeImage(self.originalSize, C.STANDARD_W, C.STANDARD_H)
        self.h, self.w = getImageSizes(self.original)
        self.grayImg = toGray(self.original)
        self.sv = toSV(self.original)
        self.em = self.trainEM(resizeImage(self.sv, 160, 120), 160, 120)
        
        self.pixelDict = {}
        self.segImg = self.getLeafPixels(self.sv, self.h, self.w, self.grayImg) #get segmentatiuon in white
        #self.segImg = cleanStem(self.segImg, self.h, self.w)
        
        
        self.newRGB = self.ApplyMaskToRGB(self.original, self.segImg, self.w, self.h)
        self.resizedLeafImg = self.grayImg*self.segImg
        
        
        #self.em = self.trainEMInjury(resizeImage(self.resizedLeafImg, 160, 120), 160, 120)
        #self.injuryMask = self.getInjuryPixels(self.resizedLeafImg, self.h, self.w, self.grayImg)
        #self.injuryMask = self.injuryMask *  255
        
        
        self.componentsImg = self.superPixels(self.newRGB)
        self.pixelDict = {}
        self.trainEMInjury()
        self.injuryByCluster = self.getInjuryPixels()
        #self.injuryMask = self.createInjuryMask(self.componentsImg);
        
        #self.resultImg = self.ApplyMaskToRGB(self.original, self.injuryMask, self.w, self.h)
        self.resultImg = self.applyInjuryMask(self.componentsImg, self.original)
        
        #self.contoursImg = normalizeLeafArea(self.original, segmentContours, area, C.STANDARD_LEAF_AREA)
        #self.finalSegImg = normalizeLeafArea(self.segImg, segmentContours, area, C.STANDARD_LEAF_AREA)
    
    #creates a mask for injuries of the leaf
    def createInjuryMask(self, componentsImg):
        h = self.h
        w = self.w
        mask = np.zeros((h,w), 'uint8')
        for y in range(0, h):
            for x in range(0, w):
                if componentsImg[y,x] in self.injuryByCluster and self.injuryByCluster[componentsImg[y,x]]==True:
                    mask[y,x] = 1
        return mask
    
    def applyInjuryMask(self, componentsImg, rgbImg):
        h = self.h
        w = self.w
        newRGB = rgbImg.copy();
        for y in range(0, h):
            for x in range(0, w):
                if componentsImg[y,x] in self.injuryByCluster and self.injuryByCluster[componentsImg[y,x]]==True:
                    newRGB[y,x] = np.array([0,0,255])
        return newRGB
                
    
    def ApplyMaskToRGB(self, img, mask, w, h):
        #mask = mask + -1
        b,g, r = cv2.split(img)
        r = r * mask
        g = g * mask
        b = b * mask
        rgbArray = np.zeros((h,w,3), 'uint8')
        rgbArray[..., 0] = b
        rgbArray[..., 1] = g
        rgbArray[..., 2] = r
        return rgbArray
                
    def ColorDistance(rgb1, rgb2):
        '''d = {} distance between two colors(3)'''
        rm = 0.5*(rgb1[0]+rgb2[0])
        d = sum((2+rm,4,3-rm)*(rgb1-rgb2)**2)**0.5
        return d

    
    def superPixels(self, img):
        segments = slic(img, n_segments = 50000, sigma = 6, convert2lab = True)
        #print(segments)
        #print(np.size(segments))
        # show the output of SLIC
        '''
        fig = plt.figure("Superpixels -- %d segments" % (numSegments))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(img, segments))
        plt.axis("off")
        '''
        x = np.asarray(segments).reshape(-1)
        y = np.bincount(x)
        ii = np.nonzero(y)[0]
        self.freqByCluster = zip(ii,y[ii])
        self.colorByCluster = {}
        #print(freqByCluster)
        
        #showImage(mark_boundaries(img, segments),"fgfg")
        #cv2.waitKey()
        
        for y in range(0, self.h):
            for x in range(0, self.w):
                key = segments[y,x]
                if img[y,x].all() > 0:
                    if not key in self.colorByCluster:
                        self.colorByCluster[key] = img[y,x]
        
        color1_rgb = sRGBColor(1.0, 1.0, 1.0)
        color1_lab = convert_color(color1_rgb, LabColor)
        self.distanceByCluster = {}
        for key, value in self.colorByCluster.iteritems():
            #rm = 0.5*abs(value[0]+white[0])
            #d = np.sum((2+rm,4,3-rm)*(value-white)**2)**0.5
            color2_rgb = sRGBColor(float(value[0])/255, float(value[1])/255, float(value[2])/255)
            color2_lab = convert_color(color2_rgb, LabColor)
            self.distanceByCluster[key] = delta_e_cie2000(color1_lab, color2_lab);
        
        return segments
        
    

    def showImages(self):
        showImage(self.original, "Original ")
        showImage(self.segImg, "Segmented Binary")
        showImage(self.resizedLeafImg, "Segmented ")
        showImage(self.resultImg, "Injury detected Img ")
        #showImage(self.injuryMask, "Injuries ")
        
        #showImage(self.shadowLessImg, "Shadowless " + self.path)
        #showImage(self.sv, "Saturation/Value " + self.path)
        #showImage(self.finalSegImg, "Segmented " + self.path)
        #showImage(self.grayImg, "Segmented " + self.path)
        #showImage(self.contoursImg, "Contour " + self.path)
        #showImage(self.venation.lbpR2P16pic, "Veins " + self.path)
        #showImage(self.venation.lbpR1P8pic, "LBPR1P8")
        #showImage(self.venation.lbpR3P16pic, "LBPR3P16")
    
    def trainEM(self, img, w, h):
        print('Training EM...')
        t = StopWatch()
        samples = []
        nhe = h
        nwe = w
        nhs = 0
        nws = 0
        for y in range(nhs, nhe):
            for x in range(nws, nwe):
                samples.append(img[y, x])#[1:2])
        tsamples = np.float32( np.vstack(samples) )
        em = EM(nclusters = 2, covMatType=cv2.EM_COV_MAT_DIAGONAL)#, cov_mat_type = cv2.EM_COV_MAT_DEFAULT)
        em.train(tsamples)#, weights0 = [0.5, 0.5])
        t.stopAndPrint()
        return em
    
    def predict(self, x, y):
        key = hash((x,y))
        if key in self.pixelDict:
            return self.pixelDict[key]
        _, probs = self.em.predict(np.float32([[x, y]]))
        self.pixelDict[key] = probs[0][0] >= probs[0][1]
        return self.pixelDict[key]
    
    def invertBitImage(self, x):
        if x==1:
            return 0
        return 1
    
    def getLeafPixels(self, img, h, w, grayImg):
        print("Predicting...")
        t = StopWatch()
        predict = np.vectorize(self.predict)
        segmentedr = predict(img[:,:,0], img[:,:,1])
        
        t.stopAndPrint()
            
        invertBitImage = np.vectorize(self.invertBitImage)
        inverted = invertBitImage(segmentedr)
        
        average =  np.average(segmentedr * grayImg)
        invAverage = np.average(inverted * grayImg)
        
        if invAverage < average:
            segmentedr = inverted
       
        segImg, _ = findLeafComponent(segmentedr, h, w)
        segImg = segImg * 255
        segImg = segImg.astype(np.uint8)        
        
        return segImg
    


    def trainEMInjury(self):
        print('Training EM for Injuries...')
        samples = self.distanceByCluster.values()
        t = StopWatch()
        tsamples = np.float32( np.vstack(samples) )
        self.em = EM(nclusters = 2, covMatType=cv2.EM_COV_MAT_DIAGONAL)#, cov_mat_type = cv2.EM_COV_MAT_DEFAULT)
        self.em.train(tsamples)#, weights0 = [0.5, 0.5])
        t.stopAndPrint()
        return self.em

    #cluster by duistance
    def predictInjury(self, distance, clusterKey):
        _, probs = self.em.predict(np.float32([distance]))
        self.pixelDict[clusterKey] = probs[0][0] >= probs[0][1]

    def getInjuryPixels(self):
        print("Predicting Injuries...")
        for key, sample in self.distanceByCluster.iteritems():
            self.predictInjury(sample, key)
        '''
        print(self.pixelDict)
        
        for value in self.pixelDict.values():
            if value==False:
                print(value)
        '''
        return self.pixelDict