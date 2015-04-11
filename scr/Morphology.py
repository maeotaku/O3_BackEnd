from StopWatch import *
import Constants as C
from Components import *
from Exceptions.CustomException import ContoursException
import numpy as np
import cv2
import os 
import math
from scipy import ndimage
#from skimage import measure
#pip install -U scikit-image



'''
def combineImages(*args):
    numImages = 0
    imageArray = []
    
    for img in args:
        if img != None and img.width > 0 and img.height > 0:
            imageArray.append(img)
            numImages += 1
    
    if numImages <= 0:
        return None
    
    # Find the largest x and y dimensions out of all the images
    # The resulting grid of images will all have this size, 
    # whether or not the images fit (though they won't be scaled).
    colWidth = max(imageArray, key=attrgetter('width')).width
    rowHeight = max(imageArray, key=attrgetter('height')).height
    
    # Square-root of the number of images will tell us how big the
    # sides of the square need to be. Ceiled to ensure they always
    # all fit.
    grid = int(math.ceil(math.sqrt(numImages)))
    
    combinedImage = cv2.CreateImage((colWidth*grid, rowHeight*grid), 8, 3)
    
    cv2.Set(combinedImage, cv2.CV_RGB(50,50,50));
    
    for index, img in enumerate(imageArray):
        # Ensure all images are same type
        if img.nChannels == 1:
            colourImg = cv2.CreateImage((img.width, img.height), 8,3)
            cv2.CvtColor(img, colourImg, cv2.CV_GRAY2RGB)
            img = colourImg
        
        # Which grid square are we up to?
        row = int(math.ceil(index / grid))
        column = index % grid
        
        cv2.SetImageROI(combinedImage, (row*colWidth, column*rowHeight, img.width, img.height))
        cv2.Copy(img, combinedImage)
        cv2.ResetImageROI(combinedImage);
    
    return combinedImage
'''

def saveImage(name, img):
    cv2.imwrite(name,img)

def loadImage(name):
    img = cv2.imread(name)
    return img

def toGray(img):
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gimg

def toHSV(img):
    gimg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return gimg

def toSV(img): 
    hsv = toHSV(img)
    #s = cleanShadows(hsv, 525, 700)
    #showImage(s, "shadows")
    #cv2.waitKey()
    #hsv[:,:,0] = 0 #works better with shadows
    #saveImage("/Users/maeotaku/Documents/H.png", hsv[:,:,0])
    #saveImage("/Users/maeotaku/Documents/S.png", hsv[:,:,1])
    #saveImage("/Users/maeotaku/Documents/V.png", hsv[:,:,2])
    #saveImage("/Users/maeotaku/Documents/HSV.png", img)
    #cv2.waitKey()
    #return hsv[:,:,2]
    
    return hsv[:,:,[1,2]] #get rid of hue

    
    #return hsv[:,:,2] #only one channel

def split_HSV(hsv_img):
    h, s, v = cv2.split(hsv_img)
    return h, s, v#hs_img

def resizeImage(img, resx, resy):
    ar = getAspectRatio(img)
    if ar > 1.0:
        img = rotateImage(img, 1)
    return cv2.resize(img, (resx, resy), interpolation=cv2.INTER_AREA)

def cutImage(img, x1, y1, x2, y2):
    return img[int(x1):int(x2+1), int(y1):int(y2+1)]
                
def normalize(histogram):
    return np.divide(histogram, np.sum(histogram))
    
def getImageSizes(img):
    if len(img.shape) == 2:
        height, width = img.shape
    else:
        height, width, _ = img.shape
    return height, width

def getAspectRatio(img):
    height, width = getImageSizes(img)
    return float(height) / float(width)

def rotateImage(image, times):
    return np.rot90(image,times)

def showImage(img, title):
    cv2.imshow(title,img) 
    
def threadsHolding(img):
    return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)

def closing(img, kernel):
    #kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

def opening(img):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)

def dilation(img):
    kernel = np.ones((1, 1), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel, iterations=1)

def topHat(img, kernel):
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

def cleanStem(segImg, h, w):
    print("Cleaning stem....")
    t = StopWatch()
    _, nComponents = ndimage.measurements.label(segImg)
    
    def getContourAspectRatio(contour):
        _,_,w,h = cv2.boundingRect(contour)
        a = cv2.contourArea(contour)
        return float(w)/float(h) * a
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(C.STEM_KERNEL_SIZE,C.STEM_KERNEL_SIZE))
    th = topHat(segImg, kernel)
    contours, _ =  cv2.findContours(th.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    n = len(contours)
    posactual = 0
    while posactual < n:
        mask = segImg.copy()
        cv2.drawContours(mask, contours[posactual],-1,0,-1)
        _, nCComponents = ndimage.measurements.label(mask)
        if nCComponents == nComponents: #eliminate small contours, mostly garbage
            contours.pop(posactual)
            n = len(contours)
        else:
            posactual+=1
    if contours == []:
        t.stopAndPrint()
        return segImg
    ars = [getContourAspectRatio(ctr) for ctr in contours]
    stem = contours[ars.index(max(ars))]
    cv2.drawContours(segImg, [stem],-1,0,-1) #deletethe stem from the segmentation
    #segImg = getBiggestComponent(segImg) * 255
    #segImg = segImg.astype(np.uint8)
    t.stopAndPrint()
    return segImg












def pointIsInImageBorder(pointh, pointw, h, w):
    if (pointh<=10 or pointh>=h-10):
        return 1
    if (pointw<=10 or pointw>=w-10):
        return 1
    return 0

def calculateIntersectionWithImageBorder(contour, h, w):
    ws = contour[:,0]
    hs = contour[:,1]
    foo = np.vectorize(pointIsInImageBorder)
    inters = np.sum(foo(hs, ws, h, w))
    #area = 0
    #if inters>0:
    #    return inters, area
    area = cv2.contourArea(contour)
    return inters, area

def cleanSmallComponents(contours):
    n = len(contours)
    posactual = 0
    while posactual < n:
        #print(contours[posactual].shape[0])
        if contours[posactual].shape[0] < C.MINIMUM_LEAF_VALID_CONTOUR_POINT_NUMBER: #eliminate small contours, mostly garbage
            contours.pop(posactual)
            n = len(contours)
        else:
            posactual+=1
    return contours

def compactContours(contours):
    aux = []
    for ctr in contours:
        for pair in ctr:
            aux = aux + [pair[0].tolist()]
    return aux 

def getContours(img):
    contours, _ =  cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours

def getBiggestContour(img):
    contours, _ =  cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(ctr) for ctr in contours]
    return [contours[areas.index(max(areas))]]

def extractContours(segImg, checkImgBorders, h, w):
    print("Extracting contours...")
    t = StopWatch()
    contours, _ =  cv2.findContours(segImg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    maxi = 0
    if checkImgBorders:
        contours = cleanSmallComponents(contours)
        if len(contours)==0:
            raise ContoursException("No contours found")
        if len(contours)==1: #if there is only one just send it back
            return [contours[0]]
        pos = -1
        posactual = 0
        maxinters = 0
        posareas = []
        for ctr in contours:
            inters, area = calculateIntersectionWithImageBorder(ctr[:,0], h, w)
            posareas.append(area)
            if ((inters>maxinters)):
                maxinters = inters
                pos = posactual
            posactual+=1
        if pos>-1:
            posareas.pop(pos)
            contours.pop(pos)
        maxi = contours[posareas.index(max(posareas))]
    else:
        areas = [cv2.contourArea(ctr) for ctr in contours]
        maxi = contours[areas.index(max(areas))]
    if maxi.shape[0] < C.MINIMUM_LEAF_VALID_CONTOUR_POINT_NUMBER:
        raise ContoursException("No valid leaf found. Contours too small, most likely segmentation error.")
    t.stopAndPrint()
    return [maxi]
        
'''        
def cutLeafSquare(img, contour):
    
    def check(n, k, maxi):
        if n+k < 0:
            return n
        if n+k > maxi:
            return n
        return n+k
    
    h, w = getImageSizes(img)
    ys = contour[:,1]
    xs = contour[:,0]
    x1 = check(min(xs), -10, w)
    x2 = check(max(xs), 10, w)
    y1 = check(min(ys), -10, h)
    y2 = check(max(ys), 10, h)
    return cutImage(img, y1, x1, y2, x2)
'''

def cutLeafSquare(img, contour):
    contour = np.array(contour)
    def check(n, k, maxi):
        if n+k < 0:
            return n
        if n+k > maxi:
            return n
        return n+k
    
    h, w = getImageSizes(img)
    ys = contour[:,1]
    xs = contour[:,0]
    x1 = check(min(xs), -10, w)
    x2 = check(max(xs), 10, w)
    y1 = check(min(ys), -10, h)
    y2 = check(max(ys), 10, h)
    return cutImage(img, y1, x1, y2, x2)


def normalizeLeafArea(img, contours, leafArea, newLeafArea):
    leafCutImg = cutLeafSquare(img, contours)
    ar = getAspectRatio(leafCutImg)
    if ar > 1.0:
        leafCutImg = rotateImage(leafCutImg, 1)
    h, w = getImageSizes(leafCutImg)
    imgArea = h * w
    #leafArea = cv2.contourArea(contours)
    newImgArea = (imgArea * newLeafArea) / leafArea
    
    wGrowth = float(w) / float(w+h)
    hGrowth = float(h) / float(w+h)
    a = wGrowth * hGrowth
    x = abs((math.sqrt(4 * a * newImgArea)) / (2*a))
    newWidth = int(wGrowth * x)
    newHeight = int(hGrowth * x)
    return resizeImage(leafCutImg, newWidth, newHeight) 



def cleanShadows(img, h, w):
    np.set_printoptions(threshold=np.nan)
    b = img.astype(np.double)
    #print(b)
    
    def log(x):
        return math.log(x)
    
    def mini(x):
        return min(x)
    
    def maxi(x):
        return max(x)
    
    def exp(x):
        return math.exp(x)
    
    def inti(x):
        return int(x)
    
    log = np.vectorize(log)
    exp = np.vectorize(exp)
    mini = np.vectorize(mini)
    maxi = np.vectorize(maxi)
    inti = np.vectorize(inti)
    
    b[b == 0]=0.001
    r_b = log(np.divide(b[:,:,0], b[:,:,2]))
    g_b = log(np.divide(b[:,:,1], b[:,:,2]))  
    
    deg=51
    rad = deg*(math.pi/180)
    #print(rad)
    inv = (math.cos(rad)*r_b) - (math.sin(rad)*g_b)
    invexp = exp(inv)
    #print(min(invexp.flatten()))
    invexp=invexp - min(invexp.flatten())
    #print(max(invexp.flatten()))
    invexp= invexp*255 / max(invexp.flatten())
    
    x= (max(g_b.flatten()) + min(g_b.flatten())) / 2
    c1 = g_b[g_b >= x]
    c2 = g_b[g_b < x]
    mc1=np.median(c1)
    mc2=np.median(c2)
    #print(x, mc1, mc2)
    logresRG = np.ones((h, w), dtype=np.double)
    logresBG = np.ones((h, w), dtype=np.double)
    lol = np.zeros((h, w), dtype=np.double)
    i=0
    while i<h:
        j=0
        while j<w:
            if g_b[i,j] >=x:
                logresRG[i,j] = inv[i,j]*math.cos(-rad) + mc1*math.sin(-rad)
                logresBG[i,j] = -inv[i,j]*math.sin(-rad) + mc1*math.cos(-rad)  
                lol[i,j] = 255          
            else:
                logresRG[i,j] = inv[i,j]*math.cos(-rad) + mc2*math.sin(-rad)
                logresBG[i,j] = -inv[i,j]*math.sin(-rad) + mc2*math.cos(-rad)
            j+=1
        i+=1                    
    resRG = exp(logresRG)
    resBG = exp(logresBG)
    rgb = np.ones((h, w, 3), dtype=np.double)
    
    B = (3*invexp) / (resRG + resBG + 1)
    B=B-min(B.flatten())
    B=B/(max(B.flatten()))
    
    R = resRG * B
    R=R-min(R.flatten())
    R=R/(max(R.flatten()))
    
    G = resBG*B
    G=G-min(G.flatten())
    G=G/(max(G.flatten()))
    
    rgb[:,:,0] = (B/2)
    rgb[:,:,1] = G
    rgb[:,:,2] = R

    
    rgb = np.array(g_b*255).astype(np.uint8) #like black and white only
    #equ = cv2.equalizeHist(rgb)
    return rgb 
    #thresh = 127
    #ret = cv2.threshold(rgb, thresh, 255, cv2.THRESH_BINARY)[1]
    #ret = cv2.adaptiveThreshold(rgb,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_OTSU,11,2)
    #return lol 