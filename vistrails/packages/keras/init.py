###############################################################################
##
## Copyright (C) 2014-2016, New York University.
## All rights reserved.
## Contact: contact@vistrails.org
##
## This file is part of VisTrails.
##
## "Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
##
##  - Redistributions of source code must retain the above copyright notice,
##    this list of conditions and the following disclaimer.
##  - Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimer in the
##    documentation and/or other materials provided with the distribution.
##  - Neither the name of the New York University nor the names of its
##    contributors may be used to endorse or promote products derived from
##    this software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
## AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
## THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
## PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
## CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
## EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
## PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
## OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
## WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
## OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
## ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
##
###############################################################################

from __future__ import division

from vistrails.core.modules.config import ModuleSettings, IPort, OPort
from vistrails.core.modules.vistrails_module import Module, ModuleError
from vistrails.core.packagemanager import get_package_manager

from .layers import _layers
from .activations import _activations
from .models import _models
from .utils import ModuleBase, KerasBase

import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence

from pandas import read_csv
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt4Agg', warn=False)

###############################################################################
# Env variables
import os
DATASET_DIR = os.path.dirname(os.path.abspath(__file__)) + '/datasets'

###############################################################################
# Utils

class Sample(Module):
    """Given an index or percentage, divide the data into two parts
    """
    _settings = ModuleSettings(namespace="utils")
    _input_ports = [("X_train", "basic:List", {"shape": "circle"}),
                    ("y_train", "basic:List", {"shape": "circle"}),
                    ("X_test", "basic:List", {"shape": "circle"}),
                    ("y_test", "basic:List", {"shape": "circle"}),
                    ("proportion", "basic:Float", {"shape": "circle"})]
    
    _output_ports = [("X_train", "basic:List", {"shape": "circle"}),
                     ("y_train", "basic:List", {"shape": "circle"}),
                     ("X_test", "basic:List", {"shape": "circle"}),
                     ("y_test", "basic:List", {"shape": "circle"})]

    
    def compute(self):
        X_train = self.get_input("X_train")
        y_train = self.get_input("y_train")
        X_test = self.get_input("X_test")
        y_test = self.get_input("y_test")

        idx = self.get_input("proportion")

        if idx > 1:
            idx = int(idx)
        else:
            idx = int(idx*len(data))

        if X_train != None:
            self.set_output("X_train", X_train[:idx])
        if y_train != None:
            self.set_output("y_train", y_train[:idx])
        if X_test != None:
            self.set_output("X_test",  X_test[:idx])
        if y_test != None:
            self.set_output("y_test",  y_test[:idx])

class SplitCol(Module):
    _input_ports = [("data", "basic:List", {"shape": "circle"}),
                    ("col", "basic:Integer", {"shape": "circle"})]
    _output_ports = [("x", "basic:List", {"shape": "circle"}),
                     ("y", "basic:List", {"shape": "circle"})]

    def compute(self):
        data = self.get_input("data").values
        col = self.get_input("col")
        X = data[:,0:8]
        Y = data[:,8]
        self.set_output("x", X)
        self.set_output("y", Y)


class ReadCSV(ModuleBase):
    """Returns pandas dataframe from CSV file
    """
    _settings = ModuleSettings(namespace="datasets")
    _input_ports = [("filepath_or_buffer", "basic:File", {"shape": "circle"}),
                    ("delimiter", "basic:String", {"shape": "circle", "defaults": [',']}),
                    ("header", "basic:Integer", {"shape": "circle", "defaults": [0]}),
                    IPort(name="parse_dates", signature="basic:String", depth=1, shape="circle"),
                    ("index_col", "basic:List", {"shape": "circle"}),
                    ("chunksize", "basic:Integer", {"shape": "circle"}),
                    ("date_parser", "basic:String", {"shape": "circle"})]

    _output_ports = [("data", "basic:List", {"shape": "circle"})]
    
    def compute(self):
        parameters = self.get_parameters()
        parameters["filepath_or_buffer"] = parameters["filepath_or_buffer"].name
        parameters["parse_dates"] = [parameters["parse_dates"]] if parameters["parse_dates"] else False
        
        if parameters["date_parser"]:
            date_parser = parameters["date_parser"]
            parameters["date_parser"] = lambda x: datetime.strptime(x, date_parser)
        
        data = read_csv(**parameters)
        print(data.head())
        self.set_output("data", data)

###############################################################################
# Example datasets

class Imdb(Module):
    """Example dataset: imdb.
    """
    _settings = ModuleSettings(namespace="datasets")
    _input_ports = [("max_review_length", "basic:Integer", {"shape": "circle", "defaults": [0]}),
                    ("top_words", "basic:Integer", {"shape": "circle", "defaults": [5000]})]
    _output_ports = [("X_train", "basic:List", {"shape": "circle"}),
                     ("y_train", "basic:List", {"shape": "circle"}),
                     ("X_test", "basic:List", {"shape": "circle"}),
                     ("y_test", "basic:List", {"shape": "circle"})]
    
    def __init__(self):
        Module.__init__(self)
        self.filename = DATASET_DIR + '/imdb.npy'

    def loadData(self):
        try:
            data = np.load(self.filename).item()
            top_words = data['top_words']
            assert top_words == self.top_words
            X_train = data['X_train']
            y_train = data['y_train']
            X_test = data['X_test']
            y_test = data['y_test']
        except:
            print('downloading data...')
            (X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb.npz", num_words=self.top_words)
            self.saveData(X_train, y_train, X_test, y_test)
        return (X_train, y_train), (X_test, y_test)
    
    def saveData(self, X_train, y_train, X_test, y_test):
        config = {
            'top_words': self.top_words,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }
        np.save(self.filename, config)

    def compute(self):    
        self.max_review_length = self.get_input("max_review_length")
        self.top_words = self.get_input("top_words")
        (X_train, y_train), (X_test, y_test) = self.loadData()

        if self.max_review_length != 0:
            X_train = sequence.pad_sequences(X_train, maxlen=self.max_review_length)
            X_test = sequence.pad_sequences(X_test, maxlen=self.max_review_length)
        
        self.set_output("X_train", X_train)
        self.set_output("y_train", y_train)
        self.set_output("X_test", X_test)
        self.set_output("y_test", y_test)        

_modules = [ModuleBase, KerasBase, Sample, Imdb, ReadCSV, SplitCol] + _models + _layers + _activations