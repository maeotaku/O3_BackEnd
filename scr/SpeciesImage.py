from Components import *
from Curvature import Curvature
from Venation import Venation
from Morphology import *
#from Log import Log
import Constants as C
from cv2 import EM

class SpeciesImage(object):

    def __init__(self, disks, circumferences, path="", speciesInfo="", species="",  inputImg=None):
        self.em = None
        #self.histogramTypes = histogramTypes
        self.path = path
        self.speciesInfo = speciesInfo
        self.species = species
        if inputImg==None:
            self.originalSize = loadImage(path)
        else:
            self.originalSize = inputImg
        self.original = resizeImage(self.originalSize, C.STANDARD_W, C.STANDARD_H)
        self.h, self.w = getImageSizes(self.original)
        self.grayImg = toGray(self.original)
        self.img = toGray(self.original)
        self.sv = toSV(self.original)
        self.em = self.trainEM(resizeImage(self.sv, 160, 120), 160, 120)
        '''
        self.segImg, contoursBig = self.getLeafPixels(self.sv, self.h, self.w, self.img) #get segmentatiuon in white
        self.resizedLeafImg = makeMultiScaleInvariant(self.img*self.segImg, contoursBig[0][:,0], 100000)
        self.contoursImg = makeMultiScaleInvariant(self.original, contoursBig[0][:,0], 100000)
        self.finalSegImg = makeMultiScaleInvariant(self.segImg, contoursBig[0][:,0], 100000)
        '''
        self.pixelDict = {}
        self.segImg, area = self.getLeafPixels(self.sv, self.h, self.w, self.img) #get segmentatiuon in white
        #showImage(self.segImg, "With Postprocessing")
        #cv2.waitKey()
        self.segImg = cleanStem(self.segImg, self.h, self.w)
        #showImage(self.segImg, "No Stem")
        #cv2.waitKey()
        
        
        
        contoursBig = getContours(self.segImg)
        segmentContours = compactContours(contoursBig)
        self.resizedLeafImg = normalizeLeafArea(self.img*self.segImg, segmentContours, area, C.STANDARD_LEAF_AREA)
        self.contoursImg = normalizeLeafArea(self.original, segmentContours, area, C.STANDARD_LEAF_AREA)
        self.finalSegImg = normalizeLeafArea(self.segImg, segmentContours, area, C.STANDARD_LEAF_AREA)
        #self.finalSegImg[self.finalSegImg > 0] = 1         
        '''
        contoursBig = getBiggestContour(self.segImg)
        self.resizedLeafImg = normalizeLeafArea(self.img*self.segImg, contoursBig[0][:,0], area, C.STANDARD_LEAF_AREA)
        self.contoursImg = normalizeLeafArea(self.original, contoursBig[0][:,0], area, C.STANDARD_LEAF_AREA)
        self.finalSegImg = normalizeLeafArea(self.segImg, contoursBig[0][:,0], area, C.STANDARD_LEAF_AREA)
        '''
        self.thumbnail = resizeImage(self.contoursImg, 100, 75)
        #contoursSmall = getBiggestContour(self.finalSegImg)
        contoursSmall = getContours(self.finalSegImg)
        cv2.drawContours(self.contoursImg, contoursSmall,-1,(0,0,255),2)
        self.curvature = Curvature(self.finalSegImg, contoursSmall, disks, circumferences)        
        self.venation = Venation(self.resizedLeafImg*self.finalSegImg)
        #self.finalSegImg = self.finalSegImg * 255

    def showImages(self):
        #showImage(self.original, "Original " + self.path)
        #showImage(self.shadowLessImg, "Shadowless " + self.path)
        #showImage(self.sv, "Saturation/Value " + self.path)
        showImage(self.finalSegImg, "Segmented " + self.path)
        showImage(self.contoursImg, "Contour " + self.path)
        #showImage(self.venation.lbpR2P16pic, "Veins " + self.path)
        #showImage(self.venation.lbpR1P8pic, "LBPR1P8")
        #showImage(self.venation.lbpR3P16pic, "LBPR3P16")
        
    
    
    def getHistogram(self, ht):
        if ht == C.HIST_LBP_R1P8:
            return self.venation.lbpR1P8
        if ht == C.HIST_LBP_R2P16:
            return self.venation.lbpR2P16
        if ht == C.HIST_LBP_R3P16:
            return self.venation.lbpR3P16
        if ht == C.HIST_LBP_R3P24:
            return self.venation.lbpR3P24
        if ht == C.HIST_LBP_R1P8_R2P16_CONCAT:
            return self.venation.lbpR1P8_R2P16
        if ht == C.HIST_LBP_R1P8_R3P16_CONCAT:
            return self.venation.lbpR1P8_R3P16
        if ht == C.HIST_LBP_R2P16_R3P16_CONCAT:
            return self.venation.lbpR2P16_R3P16
        if ht == C.HIST_HCoS:
            return self.curvature.HCoS
        if ht == C.HIST_HCoS_LBP_R1P8_R3P16_CONCAT:
            return np.float32(np.concatenate([self.venation.lbpR1P8_R3P16, self.curvature.HCoS]))
        return None
   
   

    def trainEM2(self, img, height, width):
        height, width = getImageSizes(img)
        samples = []
        for y in range(0, height):
            for x in range(0, width):
                samples.append(img[y, x])#[1:2])
        tsamples = np.float32( np.vstack(samples) )
        print(len(tsamples), tsamples[0])
        em = cv2.EM(nclusters = 2, covMatType=cv2.EM_COV_MAT_DIAGONAL)#, cov_mat_type = cv2.EM_COV_MAT_DEFAULT)
        means0=np.float32([[0.5], [0.5]])
        covs0=np.float32([[100], [0]])
        weights0=np.float32([[0.5, 0.5]])
        '''
        means0=np.float32([[0,255],[56.43057204,109.98765725]])
        covs0=np.float32([[[2.22044605e-16,0.00000000e+00],
                           [0.00000000e+00,2.22044605e-16]],
                          [[6.96442688e+02,0.00000000e+00],
                           [0.00000000e+00,2.09675317e+03]]])
        weights0=np.float32([[0.78057292, 0.21942708]])
        '''
        retval, logLikelihoods, labels, probs =  em.trainE(tsamples, means0, covs0, weights0)
        cont=0
        while cont<3:
            retval, logLikelihoods, labels, probs = em.trainM(tsamples, probs, logLikelihoods, labels)
            means0 = em.getMat('means') 
            covs0 = np.float32(em.getMatVector('covs'))
            weights0 = em.getMat('weights')
            retval, logLikelihoods, labels, probs = em.trainE(tsamples, means0, covs0, weights0, logLikelihoods, labels, probs)
            cont+=1
        return em

    
    
    def trainEM(self, img, w, h):
        print('Training EM...')
        t = StopWatch()
        samples = []
        nhe = h#int(h * 0.7)
        nwe = w#int(w * 0.7)
        nhs = 0#int(h * 0.3)
        nws = 0#int(w * 0.3)
        for y in range(nhs, nhe):
            for x in range(nws, nwe):
                samples.append(img[y, x])#[1:2])
        tsamples = np.float32( np.vstack(samples) )
        em = EM(nclusters = 2, covMatType=cv2.EM_COV_MAT_DIAGONAL)#, cov_mat_type = cv2.EM_COV_MAT_DEFAULT)
        em.train(tsamples)#, weights0 = [0.5, 0.5])
        #print("Means", em.getMat('means')) 
        #print("Covs", np.float32(em.getMatVector('covs')))
        #print("Weights", em.getMat('weights'))
        t.stopAndPrint()
        return em
    
    
    def predict(self, x, y):
        key = hash((x,y))
        if key in self.pixelDict:
            return self.pixelDict[key]
        #print(x, y, len(self.pixelDict))
        _, probs = self.em.predict(np.float32([[x, y]]))
        self.pixelDict[key] = probs[0][0] >= probs[0][1]
        #if probs[0][0] >= probs[0][1]:
        #    self.pixelDict[key] = 1
        return self.pixelDict[key]
    
    
      
    
    '''
    def predict(self, x, y):
        #print(x, y)
        _, probs = self.em.predict(np.float32([[x, y]]))
        return probs[0][0] >= probs[0][1]
            #return 1; #white
        #return 0; #black
    '''
    
    
    #it seems OpenCV has an issue with EM memory handling internally on its C code. SOmethimes even if a new instance is created, it holds 
    #data from the previous instance. I tried deleting the instances, however the issue seems to be happening internally in OpenCV implementation
    #This function is an ehuristic workaround to get a good segmentation when it coms reversed (black becomes white and white becomes black)
    #0=black, 255=white
    def invertBitImage(self, x):
        if x==1:
            return 0
        return 1
    
    def getLeafPixels(self, img, h, w, grayImg):
        print("Predicting...")
        t = StopWatch()
        '''
        segmentedr = np.zeros((h*w), dtype=np.float32())
        #6 secs
        x = 0
        for (i,j) in np.ndindex(h, w):
            segmentedr[x] = self.predict(np.float32([img[i,j]]))
            x+=1
        '''
        predict = np.vectorize(self.predict)
        segmentedr = predict(img[:,:,0], img[:,:,1])
        
        t.stopAndPrint()
            
        invertBitImage = np.vectorize(self.invertBitImage)
        inverted = invertBitImage(segmentedr)
        
        average =  np.average(segmentedr * grayImg)
        invAverage = np.average(inverted * grayImg)
        
        if invAverage < average:
            segmentedr = inverted
       
        #using components
        segImg, area = findLeafComponent(segmentedr, h, w)
        segImg = segImg * 255
        segImg = segImg.astype(np.uint8)        
        #showImage(segImg, "sdfsdfdsf")
        #cv2.waitKey()
        
        return segImg, area
        #return segImg, contoursBig

        