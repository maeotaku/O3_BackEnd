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
    es = SpeciesImage(path="/Users/maeotaku/Documents/OzoneTest4.png")    
    es.showImages()
    print(es.GetInjuryPercentage())
    #saveImageObjs("/Users/maeotaku/Documents/", "adfdsfsdf", es, [])
    print("Done") 
    cv2.waitKey()
    