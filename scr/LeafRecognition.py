from SpeciesImage import SpeciesImage
from Files import *
from Label import *
from Morphology import *
from Experiments.MultipleRuns import *
from Constants import *
from Exceptions.CustomException import ContoursException
from scipy import ndimage, interpolate, signal as sg
import cv2
import numpy as np
import os
from Disk import *

print(os.path.dirname(ndimage.__file__))
print(os.path.dirname(cv2.__file__))
print(os.path.dirname(np.__file__))
print(os.path.dirname(cv2.__file__))

if __name__ == '__main__':
    disks = generateDiskKernelsMasks(25) #used for naive
    circumferences = generateDiskCircumferenceKernelsMasks(25)
    #disks = generateDiskKernels(25) #used for complex curvature based on paper
    es = SpeciesImage(path="/Users/maeotaku/Documents/OzoneTest4.png")    
    es.showImages()
    print(es.GetInjuryPercentage())
    #saveImageObjs("/Users/maeotaku/Documents/", "adfdsfsdf", es, [])
    print("Done") 
    cv2.waitKey()
    