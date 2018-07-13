from vistrails.core.modules.vistrails_module import Module
from vistrails.core.modules.config import IPort, OPort
import numpy as np

def __init__(self):
	Module.__init__(self)


class Soma(Module):
	_input_ports = [IPort(name = "n1", signature = "basic:float"),
	IPort(name = 'n2', signature = "basic:float")]
	_output_ports = [OPort(name = "value", signature = "basic:float")]

	def compute(self):
		v1 = self.get_input("n1")
		v2 = self.get_input("n2")

		self.set_output("value", self.sum(v1,v2))

	def sum(self, v1, v2):
		return v1 + v2;


_modules = [Soma,]