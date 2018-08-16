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
from keras.layers import Dropout as KerasDropout
from keras.layers import LSTM as KerasLSTM
from keras.layers.embeddings import Embedding as KerasEmbedding
from keras.layers import Activation

from .constants import _activations_list

###############################################################################
# Layer functions
initializer_shape = [(0,0),(1,0),(0,1)]
regularizer_shape = [(0,0),(1,0),(1,1)]
constraint_shape = [(1,1),(1,0),(0,1)]


class KerasBase(Module):
    """Base class for all keras item.
    """
    _settings = ModuleSettings(abstract=True)

    _input_ports = [IPort(name="model", signature="basic:List", shape="diamond")]
    _output_ports = [("model", "basic:List", {"shape": "diamond"})]

    def gen_tuple(self, port):
        port_name, port_type = port
        if(port_type == "basic:List"):
            try:
                value = self.get_input_list(port_name)
            except:
                value = None
        else:
            try:
                value = self.get_input(port_name)
            except:
                value = None
        return (port_name, value)

    def get_parameters(self):
        input_ports = filter(lambda x: x[0] != "model", self._input_ports[:])
        input_ports = [port[0:2] for port in input_ports[:]]
        return dict(map(self.gen_tuple, input_ports))
    
    def get_model(self):
        return self.get_input("model")
    
class KerasLayer(KerasBase):
    """Base class for all keras layers.
    """
    _settings = ModuleSettings(abstract=True)
    _input_ports = [IPort(name="units", signature="basic:Integer", shape="circle"),
                    IPort(name="activation", signature="basic:String", shape="circle", entry_type="enum", values=_activations_list, default="linear"),
                    IPort(name="use_bias", signature="basic:Boolean", shape="circle", entry_type="enum", values=[True, False], default=True)]

class Dense(KerasLayer):
    """Fully-connected layer to Keras model.
    """
    _settings = ModuleSettings(namespace="layers")

    def compute(self):
        parameters = self.get_parameters()
        model = self.get_model()

        if isinstance(model, tuple):
            model, input_dim = model
            parameters["input_dim"] = input_dim
        
        model.add(KerasDense(**parameters))
        self.set_output("model", model)

class Dropout(KerasBase):
    """Fully-connected layer to Keras model.
    """
    _settings = ModuleSettings(namespace="layers")
    _input_ports = [IPort(name="rate", signature="basic:Float", shape="circle", default=0.0),
                    IPort(name="noise_shape", signature="basic:List"),
                    IPort(name="seed", signature="basic:Integer", shape="circle")]

    def compute(self):
        parameters = self.get_parameters()
        model = self.get_model()

        model.add(KerasDropout(**parameters))
        self.set_output("model", model)        

class LSTM(KerasLayer):
    """Long Short-Term Memory layer - Hochreiter 1997.
    """
    _settings = ModuleSettings(namespace="layers")

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


_layers = [KerasBase, KerasLayer, Dense, Dropout, LSTM, Embedding]