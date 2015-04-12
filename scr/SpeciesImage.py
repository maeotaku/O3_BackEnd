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

    def __init__(self, path=None, inputImg=None):
        self.injuryPercentage = 0
        self.colorByCluster = {}
        self.distanceByCluster = {}
        self.freqByCluster = {}
        self.injuryByCluster = {}
        self.path = path
        if inputImg==None and path!=None:
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
        self.newRGB = self.ApplyMaskToRGB(self.original, self.segImg, self.w, self.h)
        self.resizedLeafImg = self.grayImg*self.segImg
        
        self.componentsImg = self.superPixels(self.newRGB)
        self.pixelDict = {}
        self.trainEMInjury()
        self.injuryByCluster = self.getInjuryPixels()
        #self.injuryMask = self.createInjuryMask(self.componentsImg);
        #self.resultImg = self.ApplyMaskToRGB(self.original, self.injuryMask, self.w, self.h)
        self.resultImg = self.applyInjuryMask(self.componentsImg, self.original)

    def GetInjuryPercentage(self):
        return self.injuryPercentage
    
    def calculateInjuryPercentage(self):
        cluster1freq = 0
        cluster1min = 100
        cluster2freq = 0
        cluster2min = 100
        for key, value in self.injuryByCluster.iteritems():
            if value: #cluster1=True
                cluster1freq+=self.freqByCluster[key]
                if self.distanceByCluster[key]< cluster1min:
                    cluster1min = self.distanceByCluster[key]
            else:
                cluster2freq+=self.freqByCluster[key]
                if self.distanceByCluster[key]< cluster2min:
                    cluster2min = self.distanceByCluster[key]
        totalArea = cluster1freq + cluster2freq
        if cluster1min < cluster2min:
            self.injuryPercentage = float(cluster1freq) / float(totalArea)
            return True
        else:
            self.injuryPercentage = float(cluster2freq) / float(totalArea)
            return False
            
        
            
        
    
    def applyInjuryMask(self, componentsImg, rgbImg):
        injuryBoolean = self.calculateInjuryPercentage()
        h = self.h
        w = self.w
        newRGB = rgbImg.copy();
        for y in range(0, h):
            for x in range(0, w):
                if componentsImg[y,x] in self.injuryByCluster and self.injuryByCluster[componentsImg[y,x]]==injuryBoolean:
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
        segments = slic(img, n_segments = 1000, sigma = 10, convert2lab = True)
        x = np.asarray(segments).reshape(-1)
        y = np.bincount(x)
        ii = np.nonzero(y)[0]
        self.freqByCluster = dict(zip(ii,y))
        self.colorByCluster = {}
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
        showImage(mark_boundaries(self.original, self.componentsImg),"Segments")
        #showImage(self.injuryMask, "Injuries ")
        #showImage(self.sv, "Saturation/Value " + self.path)
        #showImage(self.finalSegImg, "Segmented " + self.path)
        #showImage(self.grayImg, "Segmented " + self.path)
    
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
        return self.pixelDict