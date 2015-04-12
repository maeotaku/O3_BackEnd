
import traceback
from SpeciesImage import SpeciesImage
import cv2
import numpy as np
import os
import random
import json
import base64
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
    return np.asarray(bytearray(f.read()), dtype=np.uint8)
     
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
        incomingData = jsonify(request.get_json(force=True))
        fileUpload = base64.decodestring(json.dumps(incomingData)['image'])
        if fileUpload:
            #file = fileUpload.file
            img = cv2.imdecode(getNPFromFile(fileUpload), cv2.CV_LOAD_IMAGE_UNCHANGED) # This is dangerous for big files
            data["damage_percentage"] = random.uniform(0,1)
            #data["InjuryDetectedImage"] = Image.fromarray(img)
            data["height"] = len(img[0])
            data["width"] = len(img)
            #s = SpeciesImage(None, img)
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