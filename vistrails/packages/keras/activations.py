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

from keras.layers import advanced_activations, Activation
from keras import activations


class KerasActivation(Module):
    """Base class for all keras Activation.
    """
    _settings = ModuleSettings(abstract=True)
    _input_ports = [("model", "basic:List", {"shape": "diamond"})]
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

class LeakyReLU(KerasActivation):
    """Softmax activation function.
    """
    _settings = ModuleSettings(namespace="activations")
    _input_ports = [("alpha", "basic:Float", {"shape": "circle", "defaults": [0.3]})]

    def compute(self):
        parameters = self.get_parameters()
        model = self.get_model()
        model.add(advanced_activations.LeakyReLU(**parameters))
        self.set_output("model", model)

class ELU(KerasActivation):
    """Elu activation function.
    """
    _settings = ModuleSettings(namespace="activations")
    _input_ports = [("alpha", "basic:Float", {"shape": "circle", "defaults": [1.0]})]

    def compute(self):
        parameters = self.get_parameters()
        model = self.get_model()
        model.add(advanced_activations.ELU(**parameters))
        self.set_output("model", model)

class ThresholdedReLU(KerasActivation):
    """ThresholdedReLU activation function.
    """
    _settings = ModuleSettings(namespace="activations")
    _input_ports = [("theta", "basic:Float", {"shape": "circle", "defaults": [1.0]})]

    def compute(self):
        parameters = self.get_parameters()
        model = self.get_model()
        model.add(advanced_activations.ThresholdedReLU(**parameters))
        self.set_output("model", model)

class Softmax(KerasActivation):
    """Softmax activation function.
    """
    _settings = ModuleSettings(namespace="activations")
    _input_ports = [("axis", "basic:Integer", {"shape": "circle", "defaults": [-1]})]

    def compute(self):
        parameters = self.get_parameters()
        model = self.get_model()
        model.add(advanced_activations.Softmax(**parameters))
        self.set_output("model", model)

class PReLU(KerasActivation):
    """PReLU activation function.
    """
    _settings = ModuleSettings(namespace="activations")
    _input_ports = [("alpha_initializer", "basic:String", {"shape": "circle", "defaults": ["zeros"]}),
                    ("alpha_regularizer", "basic:String", {"shape": "circle"}),
                    ("alpha_constraint", "basic:String", {"shape": "circle"}),
                    ("shared_axes", "basic:List", {"shape": "circle"})]

    def compute(self):
        parameters = self.get_parameters()
        model = self.get_model()
        model.add(advanced_activations.PReLU(**parameters))
        self.set_output("model", model)

_activations = [KerasActivation, LeakyReLU, PReLU, ELU, ThresholdedReLU, Softmax]