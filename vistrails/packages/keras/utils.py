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

class ModuleBase(Module):
    """Base class for all vistrails module.
    """
    _settings = ModuleSettings(abstract=True)

    def gen_tuple(self, port):
        port_name, port_type = port
        if(port_type == "basic:List"):
            try:
                value = self.get_input_list(port_name)
                value = value[0] if len(value) == 1 else value
            except:
                value = self.get_default_value(port_name)
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

class KerasBase(ModuleBase):
    """Base class for all keras item.
    """
    _settings = ModuleSettings(abstract=True)

    _input_ports = [IPort(name="model", signature="basic:List", shape="diamond")]
    _output_ports = [OPort(name="model", signature="basic:List", shape="diamond")]
    
    def get_model(self):
        return self.get_input("model")