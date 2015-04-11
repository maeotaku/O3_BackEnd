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

def createHistogramFolders(folder, histogramTypes):
    for ht in histogramTypes:
        createFolder(getPath(folder, matchHistToName(ht)))
    createFolder(getPath(folder, "_Segmented"))
    createFolder(getPath(folder, "_Contours"))
    createFolder(getPath(folder, "_LBP"))
    createFolder(getPath(folder, "_Thumbnails"))
            
def saveImageObjs(folder, imagename, sample, histogramTypes):
    for ht in histogramTypes:
        saveObj(getPath(getPath(folder,matchHistToName(ht)), imagename + ".histogram"), sample.getHistogram(ht))
    saveImage(getPath(getPath(folder,"_Segmented"), imagename + "_seg.jpg"), sample.finalSegImg)
    saveImage(getPath(getPath(folder,"_Contours"), imagename + "_contour.jpg"), sample.contoursImg)
    #saveImage(getPath(getPath(folder,"_LBP"), imagename + "_LBPR1P8.jpg"), sample.venation.lbpR1P8pic)
    saveImage(getPath(getPath(folder,"_LBP"), imagename + "_LBPR3P16.jpg"), sample.venation.lbpR3P16pic)
    #saveImage(getPath(getPath(folder,"_LBP"), imagename + "_LBPR3P16.jpg"), sample.venation.lbpR3P16pic)
    saveImage(getPath(getPath(folder,"_Thumbnails"), imagename + "_thumbnail.jpg"), sample.thumbnail)
    
      

def buildHistogramFiles(path, histogramTypes, disks, circumferences):
    speciesFolders = cleanHiddenFiles(getFolderContents(path))
    cont=1
    for folderName in speciesFolders:
        folder = getPath(path, folderName)
        if os.path.isdir(folder):
            print(folder)
            images = cleanHiddenFiles(getFolderContents(folder))
            createHistogramFolders(folder, histogramTypes)
            for imageName in images:
                try:
                    image = getPath(folder, imageName)
                    print(image)
                    imagename = getNameOfFile(image)
                    sample = SpeciesImage(disks, circumferences, path=image, species=cont, speciesInfo=imagename)
                    saveImageObjs(folder, imagename, sample, histogramTypes)
                except Exception as e:
                    #print("Error: " , imageName)
                    print(e)
            cont+=1
            


if __name__ == '__main__':
    disks = generateDiskKernelsMasks(25) #used for naive
    circumferences = generateDiskCircumferenceKernelsMasks(25)
    #disks = generateDiskKernels(25) #used for complex curvature based on paper
    es = SpeciesImage(disks, circumferences, path="/Users/maeotaku/Documents/OzoneTest.png")    
    es.showImages()
    #saveImageObjs("/Users/maeotaku/Documents/", "adfdsfsdf", es, [])
    print("Done") 
    cv2.waitKey()
    