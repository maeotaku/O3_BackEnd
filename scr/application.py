
import traceback
from SpeciesImage import SpeciesImage
import cv2
import numpy as np
import os
import random
import json
import base64
from Morphology import *
'''
from Files import *
from Label import *
from Morphology import *
from Experiments.MultipleRuns import *
from Constants import *
from FindK import *
from Data.DataSet import *
from Data.DataSetSeparate import *
from Disk import *
from Exceptions.CustomException import ContoursException
from scipy import ndimage, interpolate, signal as sg
#from bottle import route, run, template, request
import jsonpickle
import random
'''
from flask import Flask, render_template, request, url_for

application = Flask(__name__)
app = application

def getNPFromFile(f):
    #return np.asarray(bytearray(f.read()), dtype=np.uint8)
    filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as fi:
        fi.write(f)
    return loadImage(filename)
    #return np.frombuffer(f,dtype=np.float64)
    
def setFileFromNP(arr):
    #return np.asarray(bytearray(f.read()), dtype=np.uint8)
    filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
    saveImage(filename, arr)
    file = open(filename, 'r')
    stream = file.read()
    return base64.b64encode(stream)
     
application = Flask(__name__)
app = application

@app.route("/", methods=['POST'])  
def do_upload():
    data = {}    
    try:
        '''
        files = request.files
        for name in files:
            fileUpload = files[name]
        '''
        image = request.form['image']
        fileUpload = base64.decodestring(image)
        if fileUpload:
            #file = fileUpload.file
            #img = cv2.imdecode(getNPFromFile(fileUpload), cv2.CV_LOAD_IMAGE_UNCHANGED) # This is dangerous for big files
            img = getNPFromFile(fileUpload) # This is dangerous for big files
            #random.uniform(0,1)
            #data["InjuryDetectedImage"] = Image.fromarray(img)
            s = SpeciesImage(path=None, inputImg=img)
            data["damage_percentage"] = s.GetInjuryPercentage()
            data["result_image"] =  setFileFromNP(s.resultImg)
        json_data = json.dumps(data)
        return json_data

    except Exception as e:
        print(e)
        return "Exception: " + traceback.format_exc()
    return "No classification done."

if __name__ == "__main__":         
    app.run(
            host="0.0.0.0",
            port=int("8080")
    )