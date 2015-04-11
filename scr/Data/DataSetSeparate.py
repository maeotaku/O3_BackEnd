from Files import *
from Label import *
from DataSetBase import DataSetBase
import Constants as C
import random


class DataSetSeparate(DataSetBase):
    
    def __init__(self,):
        DataSetBase.__init__(self)
        
    def load(self, ht, trainingPath, testingPath):
        DataSetBase.load(self, ht, trainingPath, self.trainingData, self.trainingLabels)
        DataSetBase.load(self, ht, testingPath, self.testingData, self.testingLabels)
    '''
    def AllvsAll(self):
        aux = self.testingData + self.trainingData
        self.testingData = aux
        self.trainingData = aux
        aux = self.testingLabels + self.trainingLabels
        self.testingLabels = aux
        self.trainingLabels = aux  
    '''