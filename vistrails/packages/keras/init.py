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

from vistrails.core.modules.config import ModuleSettings
from vistrails.core.modules.vistrails_module import Module
from vistrails.core.packagemanager import get_package_manager

import numpy as np
from keras.models import Sequential as KerasSequential
from keras.layers import Dense as KerasDense
from keras.layers import Activation
from keras.datasets import imdb

###############################################################################
# Example datasets

class Imdb(Module):
    """Example dataset: imdb.
    """
    _settings = ModuleSettings(namespace="datasets")
    _output_ports = [("train", "basic:List", {'shape': 'circle'}),
                     ("test", "basic:List", {'shape': 'circle'})]

    def compute(self):
        train, test = imdb.load_data(path="imdb.npz", num_words=None, skip_top=0, maxlen=None, seed=113, start_char=1, oov_char=2, index_from=3)
        self.set_output("train", train)
        self.set_output("test", test)

###############################################################################
# Model functions

class Sequential(Module):
    """Sequential model from keras.
    """
    _settings = ModuleSettings(namespace="models")
    _input_ports = [("input_dim", "basic:Integer", {"shape": "circle"})]
    _output_ports = [("model", "basic:List", {"shape": "diamond"}),
                     ("input_dim", "basic:Integer", {"shape": "circle"})]

    def compute(self):
        model = KerasSequential()
        input_dim = self.get_input("input_dim")
        self.set_output("model", model)
        self.set_output("input_dim", input_dim)


class DenseLayer(Module):
    """Fully-connected layer to Keras model.
    """
    _settings = ModuleSettings(namespace="layers")
    _input_ports = [("model", "basic:List", {"shape": "diamond"}),
                    ("units", "basic:Integer", {"shape": "circle"}),
                    ("activation", "basic:String", {"shape": "circle", "defaults": ["relu"]}),
                    ("input_dim", "basic:Integer", {"shape": "circle", "defaults": [0]})]
    _output_ports = [("model", "basic:List", {"shape": "diamond"})]

    def compute(self):
        units = self.get_input("units")
        input_dim = self.get_input("input_dim")
        activation = self.get_input("activation")
        model = self.get_input("model")

        if input_dim != 0:
            model.add(Dense(units, activation=activation, input_dim=input_dim))
        else:
            model.add(Dense(units, activation=activation))

        self.set_output("model", model.summary())

_modules = [Sequential, DenseLayer]