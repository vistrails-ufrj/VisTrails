from vistrails.core.modules.vistrails_module import Module, ModuleError
from vistrails.core.modules.config import IPort, OPort

from nltk import sent_tokenize, word_tokenize, pos_tag


def __init__(self):
	Module.__init__(self)

class Sum(Module):
	_input_ports = [IPort(name = "n1", signature = "basic:Float"),
		IPort(name = 'n2', signature = "basic:Float")]
	_output_ports = [OPort(name = "value", signature = "basic:Float")]

	def compute(self):
		v1  = self.get_input("n1")
		v2 = self.get_input("n2")

		raw = "Hi, I'm John."
		raw = raw.strip()
		print "Raw Text" + str(raw)
		

		self.set_output("value", self.op(v1,v2))

	def op(self, v1, v2):
		return v1 + v2;

_modules = [Sum,]
