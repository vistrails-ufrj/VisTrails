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

import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential as KerasSequential
from keras.layers import Dense as KerasDense
from keras.layers import LSTM as KerasLSTM
from keras.layers.embeddings import Embedding as KerasEmbedding
from keras.layers import Activation

###############################################################################
# Env variables
import os
DATASET_DIR = os.path.dirname(os.path.abspath(__file__)) + '/datasets'

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
        print(X_train,y_train)
        if self.max_review_length != 0:
            X_train = sequence.pad_sequences(X_train, maxlen=self.max_review_length)
            X_test = sequence.pad_sequences(X_test, maxlen=self.max_review_length)
        
        self.set_output("X_train", X_train)
        self.set_output("y_train", y_train)
        self.set_output("X_test", X_test)
        self.set_output("y_test", y_test)

###############################################################################
# Preprocessing functions

class PadSequence(Module):
    """Preprocessing data to keras model
    """
    _settings = ModuleSettings(namespace="preprocessing")
    _input_ports = [("data", "basic:List", {"shape": "circle"}),
                    ("max_review_length", "basic:Integer", {"shape": "circle"})]
    _output_ports = [("data", "basic:List", {"shape": "circle"})]

    def compute(self):
        max_review_length = self.get_input("max_review_length")
        data = sequence.pad_sequences(self.get_input("data"), maxlen=max_review_length)
        self.set_output('data', data)
        

###############################################################################
# Model functions

class Sequential(Module):
    """Sequential model from keras.
    """
    _settings = ModuleSettings(namespace="models")
    _input_ports = [("input_dim", "basic:Integer", {"shape": "circle", "defaults": [0]})]
    _output_ports = [("model", "basic:List", {"shape": "diamond"}),
                     ("input_dim", "basic:Integer", {"shape": "circle"})]

    def compute(self):
        model = KerasSequential()
        input_dim = self.get_input("input_dim")
        self.set_output("model", model)
        self.set_output("input_dim", input_dim)

class Compile(Module):
    """Compile model before train.
    """
    _settings = ModuleSettings(namespace="models")
    _input_ports = [("model", "basic:List", {"shape": "diamond"}),
                    ("optimizer", "basic:String", {"shape": "circle"}),
                    ("loss", "basic:String", {"shape": "circle"}),
                    ("metrics", "basic:List", {"shape": "circle"})]
    _output_ports = [("model", "basic:List", {"shape": "diamond"})]

    def compute(self):
        optimizer = self.get_input("optimizer")
        loss = self.get_input("loss")
        metrics = self.get_input("metrics")
        model = self.get_input("model")
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.set_output("model", model)

class Fit(Module):
    """Train the compiled model.
    """
    _settings = ModuleSettings(namespace="models")
    _input_ports = [("model", "basic:List", {"shape": "diamond"}),
                    ("data", "basic:List", {"shape": "circle"}),
                    ("labels", "basic:List", {"shape": "circle"}),
                    ("epochs", "basic:Integer", {"shape": "circle"}),
                    ("batch_size", "basic:Integer", {"shape": "circle", "defaults": [32]})]
    _output_ports = [("model", "basic:List", {"shape": "diamond"})]

    def compute(self):
        data = self.get_input("data")
        labels = self.get_input("labels")
        batch_size = self.get_input("batch_size")
        epochs = self.get_input("epochs")
        model = self.get_input("model")
        print('das',data)
        model.fit(data, labels, epochs=epochs, batch_size=batch_size)
        self.set_output("model", model)

###############################################################################
# Layer functions

class Dense(Module):
    """Fully-connected layer to Keras model.
    """
    _settings = ModuleSettings(namespace="layers")
    _input_ports = [("model", "basic:List", {"shape": "diamond"}),
                    ("units", "basic:Integer", {"shape": "circle"}),
                    ("activation", "basic:String", {"shape": "circle", "defaults": ["nothing"]}),
                    ("input_dim", "basic:Integer", {"shape": "circle", "defaults": [0]})]
    _output_ports = [("model", "basic:List", {"shape": "diamond"})]

    def compute(self):
        units = self.get_input("units")
        input_dim = self.get_input("input_dim")
        activation = self.get_input("activation")
        model = self.get_input("model")

        if input_dim != 0:
            model.add(KerasDense(units=units, input_dim=input_dim))
        else:
            model.add(KerasDense(units=units))

        if activation != "nothing":
            model.add(Activation(activation))

        self.set_output("model", model)

class LSTM(Module):
    """Fully-connected layer to Keras model.
    """
    _settings = ModuleSettings(namespace="layers")
    _input_ports = [("model", "basic:List", {"shape": "diamond"}),
                    ("units", "basic:Integer", {"shape": "circle"}),
                    ("activation", "basic:String", {"shape": "circle", "defaults": ["nothing"]}),
                    ("input_dim", "basic:Integer", {"shape": "circle", "defaults": [0]})]
    _output_ports = [("model", "basic:List", {"shape": "diamond"})]

    def compute(self):
        units = self.get_input("units")
        input_dim = self.get_input("input_dim")
        activation = self.get_input("activation")
        model = self.get_input("model")

        if input_dim != 0:
            model.add(KerasLSTM(units=units, input_dim=input_dim))
        else:
            model.add(KerasLSTM(units=units))

        if activation != "nothing":
            model.add(Activation(activation))

        self.set_output("model", model)

class Embedding(Module):
    """Embedding layer to Keras model.
    """
    _settings = ModuleSettings(namespace="layers")
    _input_ports = [("model", "basic:List", {"shape": "diamond"}),
                    ("top_words", "basic:Integer", {"shape": "circle"}),
                    ("embedding_vector_length", "basic:Integer", {"shape": "circle"}),
                    ("input_length", "basic:Integer", {"shape": "circle"})]
    _output_ports = [("model", "basic:List", {"shape": "diamond"})]

    def compute(self):
        top_words = self.get_input("top_words")
        embedding_vector_length = self.get_input("embedding_vector_length")
        input_length = self.get_input("input_length")
        model = self.get_input("model")

        model.add(KerasEmbedding(top_words, embedding_vector_length, input_length=input_length))
        self.set_output("model", model)

_modules = [Imdb, PadSequence, Sequential, Compile, Fit, Dense, LSTM, Embedding]