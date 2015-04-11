'''
import traceback
from SpeciesImage import SpeciesImage
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
import cv2
import numpy as np
import os
#from bottle import route, run, template, request
import jsonpickle
import random
'''
from flask import Flask, render_template, request, url_for

application = Flask(__name__)
app = application

@app.route("/")     
def hello():         
    return "Hello World!"

if __name__ == "__main__":         
    app.run(
            host="0.0.0.0",
            port=int("8080")
    )