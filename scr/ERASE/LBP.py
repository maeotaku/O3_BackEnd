from Histogram import *
from Logging import Log
import numpy as np
import math


class LBP(object):

    def __init__(self, img, pixels, R, P, lblDict):
        self.pixels = pixels
        self.n = len(pixels)
        self.height = len(img)
        self.width = len(img[0])
        self.img = img
        self.R = R
        self.P = P
        self.lblDict = lblDict
        self.lbpImg, self.histValues = self.getUniformCircularLBPImgFAST()
        self.hist, self.bin_edges = getHist(self.histValues, 2**P)
        
    def bilinear_interpolation(self, x, y, img):
        x1, y1 = int(x), int(y)
        x2, y2 = math.ceil(x), math.ceil(y)
    
        r1 = (x2 - x) / (x2 - x1) * self.get_pixel_else_0(img, x1, y1) + (x - x1) / (x2 - x1) * self.get_pixel_else_0(img, x2, y1)
        r2 = (x2 - x) / (x2 - x1) * self.get_pixel_else_0(img, x1, y2) + (x - x1) / (x2 - x1) * self.get_pixel_else_0(img, x2, y2)
    
        return (y2 - y) / (y2 - y1) * r1 + (y - y1) / (y2 - y1) * r2    
    
    
    
    def get_pixel_else_0(self, image, idx, idy):
        if idx < int(len(image)) - 1 and idy < len(image[0]):
            return image[idx,idy]
        else:
            return 0
    
   
    
    def find_variations(self,pixel_values):
        prev = pixel_values[-1]
        t = 0
        res = 0
        for p in range(0, len(pixel_values)):
            cur = pixel_values[p]
            if cur != prev:
                t += 1
                if t>2:
                    return -1
            prev = cur
            res += pixel_values[p] * 2 ** p
        return res
    
    def thresholded(self, center, pixels):
        out = []
        for a in pixels:
            if a < center:
                out.append(1)
            else:
                out.append(0)
        return out
    
    '''
    def find_thresholded_variations(self,pixel_values, center):
        prev = pixel_values[-1]
        t = 0
        res = 0
        for p in range(0, len(pixel_values)):
            cur = pixel_values[p]
            if cur != prev:
                t += 1
                if t>2:
                    return -1
            prev = cur
            res += pixel_values[p] * 2 ** p
        return res
    '''
    
            
    
    def getUniformCircularLBPImgFAST(self):
        l = Log()
        self.hist = []
        transformed_img = np.zeros((self.height,self.width))
        for p in self.pixels:
            x=p[0]
            y=p[1]
            center = self.img[x,y]
            pixels = []
            for point in range(1, self.P + 1):
                r = x + self.R * math.cos(2 * math.pi * point / self.P)
                c = y - self.R * math.sin(2 * math.pi * point / self.P)
                if r < 0 or c < 0:
                    pixels.append(0)
                    continue            
                if int(r) == r:
                    if int(c) != c:
                        c1 = int(c)
                        c2 = math.ceil(c)
                        w1 = (c2 - c) / (c2 - c1)
                        w2 = (c - c1) / (c2 - c1)
                                        
                        pixels.append(int((w1 * self.get_pixel_else_0(self.img, int(r), int(c)) + \
                                       w2 * self.get_pixel_else_0(self.img, int(r), math.ceil(c))) / (w1 + w2)))
                    else:
                        pixels.append(self.get_pixel_else_0(self.img, int(r), int(c)))
                elif int(c) == c:
                    r1 = int(r)
                    r2 = math.ceil(r)
                    w1 = (r2 - r) / (r2 - r1)
                    w2 = (r - r1) / (r2 - r1)                
                    pixels.append((w1 * self.get_pixel_else_0(self.img, int(r), int(c)) + \
                                   w2 * self.get_pixel_else_0(self.img, math.ceil(r), int(c))) / (w1 + w2))
                else:
                    pixels.append(self.bilinear_interpolation(r, c, self.img))
            values = self.thresholded(center, pixels)
            res = self.lblDict[tuple(values)]#self.find_variations(values)
            '''
            if variations:
                res = 0
                for a in range(0, len(values)):
                    res += values[a] * 2 ** a
                transformed_img.itemset((x,y), res) 
                self.hist.append(res)
            '''
            if res > -1:
                transformed_img.itemset((x,y), res) 
                self.hist.append(res)
            
        l.printLog('Duration UniformCircularLBP:')
        return transformed_img, self.hist


    def getBasicLBPImgFAST(self):
        l = Log()
        transformed_img = np.zeros((self.height,self.width))
        for p in self.pixels:
            y = p[0]
            x = p[1]
            try:
                center = self.img[y,x]
                top_left      = 1 if self.img[y-1,x-1] < center else 0
                top_up        = 2 if self.img[y,x-1] < center else 0
                top_right     = 4 if self.img[y+1,x-1] < center else 0
                right         = 8 if self.img[y+1,x] < center else 0
                left          = 16 if self.img[y-1,x] < center else 0
                bottom_left   = 32 if self.img[y-1,x+1] < center else 0
                bottom_right  = 64 if self.img[y+1,x+1] < center else 0
                bottom_down   = 128 if self.img[y,x+1] < center else 0
        
                res = top_left + top_up + top_right + right + bottom_right + bottom_down + bottom_left + left
                transformed_img.itemset((y,x), res)
            except:
                pass
        l.printLog('Duration BasicLBP:')
        return transformed_img, None