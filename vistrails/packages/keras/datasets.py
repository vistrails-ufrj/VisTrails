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

from vistrails.core.modules.config import ModuleSettings, IPort, OPort, CIPort
from vistrails.core.modules.vistrails_module import Module, ModuleError
from vistrails.core.packagemanager import get_package_manager

from pandas import read_csv
from datetime import datetime
import numpy as np

from .utils import ModuleBase, PandasGenerator

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
        self.set_output("data", data)

class DataClassificationPreprocessing(ModuleBase):
    """Receive pandas dataframe and prepare data to classification problems
    """
    _settings = ModuleSettings(namespace="datasets")
    _input_ports = [IPort(name="data", signature="basic:List", shape="circle"),
                    IPort(name="percent_filter", signature="basic:Float", shape="circle", default=0.5),
                    IPort(name="label_column", signature="basic:Integer", shape="circle", default=-1)]
    _output_ports = [OPort(name="x", signature="basic:List", shape="circle"),
                     OPort(name="y", signature="basic:List", shape="circle")]

    def compute(self):
        parameters = self.get_parameters()
        data, percent_filter = parameters["data"], parameters["percent_filter"]
        pd_iter = PandasGenerator(data, percent_filter)
        print(next(pd_iter))




_datasets = [ReadCSV, DataClassificationPreprocessing]