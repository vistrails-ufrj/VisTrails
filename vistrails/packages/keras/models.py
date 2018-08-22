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

from .utils import KerasBase
from .constants import _optimizers_list, _losses_list, _metrics_list

from keras.models import Sequential as KerasSequential
from keras.models import model_from_json

###############################################################################
# Model functions

class Sequential(Module):
    """Sequential model from keras.
    """
    _settings = ModuleSettings(namespace="models")
    _input_ports = [("input_shape", "basic:List", {"shape": "circle", "labels": ["Dim"]})]
    _output_ports = [("model", "basic:List", {"shape": "diamond"})]

    def compute(self):
        model = KerasSequential()
        input_shape = self.get_input_list("input_shape")
        self.set_output("model", (model, input_shape))

class Compile(KerasBase):
    """Compile model before train.
    """
    _settings = ModuleSettings(namespace="models")
    _input_ports = [IPort(name="optimizer", signature="basic:String", shape="circle", entry_type="enum", values=_optimizers_list),
                    IPort(name="loss", signature="basic:String", shape="circle", entry_type="enum", values=_losses_list),
                    IPort(name="metrics", signature="basic:String", depth=1, shape="circle", entry_type="enum", values=_metrics_list)]

    def compute(self):
        parameters = self.get_parameters()
        model = self.get_model()
        model.compile(**parameters)
        self.set_output("model", model)


class Fit(KerasBase):
    """Train the compiled model.
    """
    _settings = ModuleSettings(namespace="models")
    _input_ports = [("x", "basic:List", {"shape": "circle"}),
                    ("y", "basic:List", {"shape": "circle"}),
                    ("validation_split", "basic:Float", {"shape": "circle", "defaults": [0.0]}),
                    ("epochs", "basic:Integer", {"shape": "circle"}),
                    ("batch_size", "basic:Integer", {"shape": "circle", "defaults": [32]})]

    def compute(self):
        parameters = self.get_parameters()
        model = self.get_model()

        callbacks = model.fit(**parameters)
        # summarize history for accuracy
        # plt.plot(callbacks.history['acc'])
        # if split != 0.0:
        #     plt.plot(callbacks.history['val_acc'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()

        self.set_output("model", model)

class Evaluate(Module):
    """Train the compiled model.
    """
    _settings = ModuleSettings(namespace="models")
    _input_ports = [("model", "basic:List", {"shape": "diamond"}),
                    ("data", "basic:List", {"shape": "circle"}),
                    ("labels", "basic:List", {"shape": "circle"}),
                    ("batch_size", "basic:Integer", {"shape": "circle", "defaults": [32]})]
    _output_ports = [("score", "basic:List", {"shape": "square"}),
                     ("accuracy", "basic:List", {"shape": "square"})]

    def compute(self):
        data = self.get_input("data")
        labels = self.get_input("labels")
        batch_size = self.get_input("batch_size")
        model = self.get_input("model")

        score, acc = model.evaluate(data, labels, batch_size=batch_size)

        self.set_output("score", score)
        self.set_output("accuracy", acc)


class SaveModel(Module):
    """Save model after train.
    """
    _settings = ModuleSettings(namespace="models")
    _input_ports = [IPort(name="model", signature="basic:List", shape="diamond"),
                    IPort(name="filepath", signature="basic:OutputPath", shape="circle", default="model")]

    def compute(self):
        filepath = self.get_input("filepath").name
        model = self.get_input("model")

        model_json = model.to_json()
        with open(filepath + ".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(filepath + ".h5")

class LoadModel(Module):
    """Load trained model.
    """
    _settings = ModuleSettings(namespace="models")
    _input_ports = [IPort(name="structure_model", signature="basic:File", shape="circle", default="model.json"),
                    IPort(name="weigths", signature="basic:File", shape="circle", default="model.h5")]
    _output_ports = [OPort(name="model", signature="basic:List", shape="diamond")]

    def compute(self):
        json_path = self.get_input("structure_model").name
        h5_path = self.get_input("weigths").name
        
        json_file = open(json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(h5_path)

        self.set_output("model", loaded_model)




_models = [Sequential, Compile, Fit, Evaluate, SaveModel, LoadModel]