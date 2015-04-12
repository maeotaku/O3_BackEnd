'''
Created on Apr 12, 2015

@author: maeotaku
'''
import json

def loadOzoneDB():
    with open('Data/OzoneDB.json') as data_file:    
        return json.load(data_file) 

def convertGeoToDecimal(lat, lon):
    return (lat+90)*180+lon

def jsonDBtoDict(ozoneDB):
    newOzoneDict = {}
    for di in ozoneDB:
        try:
            newOzoneDict[convertGeoToDecimal(float(di["latitude"]), float(di["longitude"]))] = float(di["DU"])
        except: # catch *all* exceptions
            pass
    return newOzoneDict

def searchFactor(lat, lon, ozoneDict, ozoneDictKeys):
    query = convertGeoToDecimal(float(lat), float(lon))
    for value in ozoneDictKeys:
        print(ozoneDict[value])
        if query <= value:
            return ozoneDict[value]
    return -1 