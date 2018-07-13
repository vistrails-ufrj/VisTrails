from vistrails.core.modules.vistrails_module import Module
from vistrails.core.modules.basic_modules import Path

from vistrails.core.modules.config import IPort, OPort
import numpy as np
import os

from vistrails.core.system import list2cmdline
import vistrails.core.requirements

def package_requirements():
	if vistrails.core.requirements.python_module_exists('keras'):
		print ("Keras is already installed and ready to use")
	else:
		raise vistrails.core.requirements.MissingRequirement('keras')

	if vistrails.core.requirements.python_module_exists('opencv-python'):
		print ("OpenCV is already installed and ready to use")
	else:
		raise vistrails.core.requirements.MissingRequirement('opencv-python')

def __init__(self):
	Module.__init__(self)


class imgLoader(Module):
	_input_ports = [IPort(name = "images-csv", signature = "basic:Path"),
	IPort(name = 'conf', signature = "basic:Path")]
	_output_ports = [OPort(name = "imgLoaded", signature = "imgLoader")]

	def compute(self):
		'''
		cmd = ['sudo','python','-m','keras']
		cmdline = list2cmdline(cmd)
		print("You're about to download all the Keras package. This may take some minutes")
		print '_DEBUG' + str(cmdline)


		if (os.system(cmdline)):
			print "Done"
		else:
			raise ModuleError(self, "Execution Failed")

		cmd = ['sudo','python','-m','cv2']
		cmdline = list2cmdline(cmd)
		print("\nYou're about to download all the OpenCV package. This may take some minutes\n")
		print '_DEBUG' + str(cmdline)


		if (os.system(cmdline)):
			print "Done"
		else:
			raise ModuleError(self, "Execution Failed")
		'''
		import dataset
		import dataloader
		

		v1 = self.get_input("images-csv")
		v2 = self.get_input("conf")
		print("")
		img_label_path = Path.translate_to_string(v1)
		print(img_label_path)
		conf_path = Path.translate_to_string(v2)
		print(conf_path)
		 
		img_dataset = dataset.ImageDataset(img_label_path,hot_encode_labels=True)
		img_dataloader = dataloader.ImageDataLoader(img_dataset, conf_path)

		self.set_output("imgLoaded", img_dataloader)

	def sum(self, v1, v2):
		return v1 + v2;


_modules = [imgLoader,]