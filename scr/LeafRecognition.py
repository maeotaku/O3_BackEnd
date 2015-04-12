from SpeciesImage import SpeciesImage
from Files import *
from Morphology import *
from Constants import *
from Exceptions.CustomException import ContoursException
from scipy import ndimage, interpolate, signal as sg
import cv2
import numpy as np
import os
from Disk import *

if __name__ == '__main__':
    lol = ["/Users/maeotaku/Documents/OzoneTest.png","/Users/maeotaku/Documents/OzoneTest2.png", "/Users/maeotaku/Documents/OzoneTest3.png", "/Users/maeotaku/Documents/OzoneTest4.png", "/Users/maeotaku/Documents/OzoneTest5.png", "/Users/maeotaku/Documents/OzoneTest6.png", "/Users/maeotaku/Documents/OzoneTest7.png"]
    for name in lol:
        es = SpeciesImage(path=name)    
        es.showImages()
        print(es.GetInjuryPercentage())
        cv2.waitKey()
    #saveImageObjs("/Users/maeotaku/Documents/", "adfdsfsdf", es, [])
    print("Done") 
    
    