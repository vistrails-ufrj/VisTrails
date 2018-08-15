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

from keras.layers import Dense as KerasDense
from keras.layers import LSTM as KerasLSTM
from keras.layers.embeddings import Embedding as KerasEmbedding
from keras.layers import Activation

###############################################################################
# Layer functions

class Dense(Module):
    """Fully-connected layer to Keras model.
    """
    _settings = ModuleSettings(namespace="layers")
    _input_ports = [("model", "basic:List", {"shape": "diamond"}),
                    ("units", "basic:Integer", {"shape": "circle"}),
                    ("activation", "basic:String", {"shape": "circle", "defaults": ["nothing"]})]
    _output_ports = [("model", "basic:List", {"shape": "diamond"})]

    def compute(self):
        units = self.get_input("units")
        activation = self.get_input("activation")
        model = self.get_input("model")
        
        if isinstance(model, tuple):
            model, input_dim = model
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
                    ("activation", "basic:String", {"shape": "circle", "defaults": ["nothing"]})]
    _output_ports = [("model", "basic:List", {"shape": "diamond"})]

    def compute(self):
        units = self.get_input("units")
        activation = self.get_input("activation")
        model = self.get_input("model")
        if isinstance(model, tuple):
            model, input_dim = model
            model.add(KerasLSTM(units=units, input_shape=(None,input_dim)))
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
                    ("output_dim", "basic:Integer", {"shape": "circle"}),
                    ("input_length", "basic:Integer", {"shape": "circle"})]
    _output_ports = [("model", "basic:List", {"shape": "diamond"})]

    def compute(self):
        output_dim = self.get_input("output_dim")
        input_length = self.get_input("input_length")
        model = self.get_input("model")
        if isinstance(model, tuple):
            model, input_dim = model
            model.add(KerasEmbedding(input_dim, output_dim, input_length=input_length))
            self.set_output("model", model)


_layers = [Dense, LSTM, Embedding]