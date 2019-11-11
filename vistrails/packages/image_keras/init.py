from vistrails.core.modules.vistrails_module import Module
from vistrails.core.modules.basic_modules import Path

from vistrails.core.modules.config import IPort, OPort
import numpy as np
import os
from PIL import Image
from vistrails.core.modules.config import ModuleSettings
from vistrails.core.system import list2cmdline
import vistrails.core.requirements
from keras.preprocessing.image import ImageDataGenerator
import keras
from matplotlib import pyplot as plt
from IPython.display import clear_output

def __init__(self):
	Module.__init__(self)



#Basics modules
class model(Module):

	_settings = ModuleSettings(abstract=True)
	_output_ports = [("genericModel", "model", {'shape': 'diamond'})]

class ImageDataGenerator(Module):

	_settings = ModuleSettings(abstract=True)
	_output_ports = [("genericLoader", "ImageDataGenerator", {'shape': 'circle'})]


#Loader Module
class imgLoader(Module):
	_settings = ModuleSettings(namespace="DataLoader")
	_input_ports = [('img_Path', "basic:Path",{'shape': 'circle'}),
	('img_sizeTupleFirstNumber','basic:Integer',{'shape': 'circle'}),
	('img_sizeTupleSecondNumber','basic:Integer',{'shape': 'circle'}),
	('batchSize','basic:Integer',{'shape': 'circle'}),
	('augmentationConf',"basic:Dictionary",{'optional': True}),
	('preprocessingConf',"basic:Dictionary",{'optional': True}),
	]
	
	_output_ports = [OPort(name = "imgLoaded", signature = "ImageDataGenerator")]

	def compute(self):
		from keras.preprocessing.image import ImageDataGenerator
		from PIL import Image


		augDict = {}
		preprocDict = {}

		if 'augmentationConf' in self.inputPorts:
			augDict = self.get_input("augmentationConf")

		if 'preprocessingConf' in self.inputPorts:
			preprocDict = self.get_input("preprocessingConf")			

		datagen = ImageDataGenerator()

		if len(augDict) > 0:
			print("Augmentation: ")
			for item in augDict:	
				print str(item)+ ': ' + str(augDict[item])
				setattr(datagen,item,augDict[item]) 
			print("")


		if len(preprocDict) > 0:
			print("Pre-Processing: ")
			for item in preprocDict:
				print str(item) + ': '+ str(preprocDict[item])				
				setattr(datagen,item,preprocDict[item]) 
			print("")		


		p1 = self.get_input("img_Path")
		img_Path = Path.translate_to_string(p1)		

		f = self.get_input('img_sizeTupleFirstNumber')
		s = self.get_input('img_sizeTupleSecondNumber')
		batch = self.get_input('batchSize')
		
		img_generator = datagen.flow_from_directory(
        img_Path,
        target_size=(f,s),
        batch_size=batch)

		print("")
	
		
		

		self.set_output("imgLoaded", img_generator)


#Param conf modules
class augmentationConf(Module):
	_settings = ModuleSettings(namespace="DataLoader")
	_input_ports = [('rotation_range',"basic:Integer",{'optional': True}),
	('width_shift_range',"basic:Float",{'optional': True}),
	('height_shift_range',"basic:Float",{'optional': True}),
	('shear_range',"basic:Float",{'optional': True}),	
	('channel_shift_range',"basic:Float",{'optional': True}),
	('fill_mode',"basic:String",{'optional': True}),
	('cval',"basic:Float",{'optional': True}),	
	('horizontal_flip',"basic:Boolean",{'optional': True}),	
	('vertical_flip',"basic:Boolean",{'optional': True})	
	]
	
	_output_ports = [OPort(name = "augDict", signature = "basic:Dictionary")]

	def compute(self):
		augDict = {}
 
		for item in self.inputPorts:
			augDict[item] =  self.get_input(item)

	
		self.set_output("augDict", augDict)


class preprocessingConf(Module):
	_settings = ModuleSettings(namespace="DataLoader")
	_input_ports = [('featurewise_center',"basic:Boolean",{'optional': True}),
	('featurewise_std_normalization',"basic:Boolean",{'optional': True}),
	('samplewise_center',"basic:Boolean",{'optional': True}),
	('samplewise_std_normalization',"basic:Boolean",{'optional': True}),
	('zca_whitening',"basic:Boolean",{'optional': True}),
	('zca_epsilon',"basic:Float",{'optional': True}),
	('rescale',"basic:Float",{'optional': True}),	
	]

	_output_ports = [OPort(name = "preprocDict", signature = "basic:Dictionary")]
	
	def compute(self):
		preprocDict = {}
 		
 		
		for item in self.inputPorts: 
			preprocDict[item] =  self.get_input(item)


		self.set_output("preprocDict", preprocDict)

#Visualization Module
class imageVisualization(Module):
	_input_ports = [IPort(name = "imgLoader", signature = "ImageDataGenerator")]

	def compute(self):
		import matplotlib.pyplot as plt

		loader = self.get_input("imgLoader")
		x_batch, y_batch = next(loader)


		print(x_batch.shape)
		idxDict = loader.class_indices
	
		plotText = ''
		for l,i in sorted(idxDict.items()):
			plotText = plotText + str(l) + ' = ' + str(i) + '\n'

		numSamples = x_batch.shape[0]
		fig = plt.figure()

		
		plt.gcf().text(0.02, 0.55, plotText, fontsize=12, style='italic',
        bbox={'facecolor':'grey', 'alpha':0.1, 'pad':10})

		for i in range (0,numSamples):
			image = x_batch[i]
			title = y_batch[i].argmax()
			title = "y: " + str(title)
			a = fig.add_subplot(3, np.ceil(numSamples/float(3)), i + 1)
			a.set_title(title)
			a.set_xticks([])
			a.set_yticks([])
			a.grid(False)
			plt.imshow(image)

		plt.subplots_adjust(left=0.2)
		mng = plt.get_current_fig_manager()
		mng.window.showMaximized()
		plt.show()


#Optimizers Modules
class sgdOptimization(Module):
	_settings = ModuleSettings(namespace="Optimizers")
	_input_ports = [IPort(name = "learningRate", signature = "basic:Float"),
	IPort(name = "momentum", signature = "basic:Float"),
	IPort(name = "decay", signature = "basic:Float"),
	IPort(name = "nesterov", signature = "basic:Boolean")]

	_output_ports = [OPort(name = "optDict", signature = "basic:Dictionary")]

	def compute(self):
		optDict= {}
		optDict['name'] = 'SGD'
		for item in self.inputPorts:
			optDict[item] =  self.get_input(item)
	
		self.set_output("optDict", optDict)


class rmsPropOptimization(Module):
	_settings = ModuleSettings(namespace="Optimizers")
	_input_ports = [IPort(name = "learningRate", signature = "basic:Float"),
	IPort(name = "rho", signature = "basic:Float"),
	IPort(name = "decay", signature = "basic:Float"),
	IPort(name = "epsilon", signature = "basic:Float")]

	_output_ports = [OPort(name = "optDict", signature = "basic:Dictionary")]

	def compute(self):
		optDict= {}
		optDict['name'] = 'RMSprop'
		for item in self.inputPorts:
			optDict[item] =  self.get_input(item)
	
		self.set_output("optDict", optDict)

class adagradOptimization(Module):
	_settings = ModuleSettings(namespace="Optimizers")
	_input_ports = [IPort(name = "learningRate", signature = "basic:Float"),
	IPort(name = "decay", signature = "basic:Float"),
	IPort(name = "epsilon", signature = "basic:Float")]

	_output_ports = [OPort(name = "optDict", signature = "basic:Dictionary")]

	def compute(self):
		optDict= {}
		optDict['name'] = 'Adagrad'
		for item in self.inputPorts:
			optDict[item] =  self.get_input(item)
	
		self.set_output("optDict", optDict)


class adadeltaOptimization(Module):
	_settings = ModuleSettings(namespace="Optimizers")
	_input_ports = [IPort(name = "learningRate", signature = "basic:Float"),
	IPort(name = "rho", signature = "basic:Float"),
	IPort(name = "decay", signature = "basic:Float"),
	IPort(name = "epsilon", signature = "basic:Float")]

	_output_ports = [OPort(name = "optDict", signature = "basic:Dictionary")]

	def compute(self):
		optDict= {}
		optDict['name'] = 'Adadelta'
		for item in self.inputPorts:
			optDict[item] =  self.get_input(item)
	
		self.set_output("optDict", optDict)

class adamOptimization(Module):
	_settings = ModuleSettings(namespace="Optimizers")
	_input_ports = [IPort(name = "learningRate", signature = "basic:Float"),
	IPort(name = "beta_1", signature = "basic:Float"),
	IPort(name = "beta_2", signature = "basic:Float"),
	IPort(name = "amsgrad", signature = "basic:Boolean"),	
	IPort(name = "decay", signature = "basic:Float"),
	IPort(name = "epsilon", signature = "basic:Float")]

	_output_ports = [OPort(name = "optDict", signature = "basic:Dictionary")]

	def compute(self):
		optDict= {}
		optDict['name'] = 'Adam'
		for item in self.inputPorts:
			optDict[item] =  self.get_input(item)
	
		self.set_output("optDict", optDict)

class adamaxOptimization(Module):
	_settings = ModuleSettings(namespace="Optimizers")
	_input_ports = [IPort(name = "learningRate", signature = "basic:Float"),
	IPort(name = "decay", signature = "basic:Float"),
	IPort(name = "beta_1", signature = "basic:Float"),
	IPort(name = "beta_2", signature = "basic:Float"),
	IPort(name = "epsilon", signature = "basic:Float")]

	_output_ports = [OPort(name = "optDict", signature = "basic:Dictionary")]

	def compute(self):
		optDict= {}
		optDict['name'] = 'Adamax'
		for item in self.inputPorts:
			optDict[item] =  self.get_input(item)
	
		self.set_output("optDict", optDict)

class nadamOptimization(Module):
	_settings = ModuleSettings(namespace="Optimizers")
	_input_ports = [IPort(name = "learningRate", signature = "basic:Float"),
	IPort(name = "beta_1", signature = "basic:Float"),
	IPort(name = "beta_2", signature = "basic:Float"),
	IPort(name = "schedule_decay", signature = "basic:Float"),
	IPort(name = "epsilon", signature = "basic:Float")]

	_output_ports = [OPort(name = "optDict", signature = "basic:Dictionary")]

	def compute(self):
		optDict= {}
		optDict['name'] = 'Nadam'
		for item in self.inputPorts:
			optDict[item] =  self.get_input(item)
	
		self.set_output("optDict", optDict)


#Model modules

class resnet50(Module):
	_settings = ModuleSettings(namespace="Models")
	_input_ports = [IPort(name = "secondNumberShape", signature = "basic:Integer"),
	IPort(name = "firstNumberShape", signature = "basic:Integer"),
	IPort(name = "numClass", signature = "basic:Integer"),
	IPort(name = "metrics", signature = "basic:String"),
	IPort(name = "loss", signature = "basic:String"),
	IPort('optimizationDict',"basic:Dictionary"),
	IPort('grayscale',"basic:Boolean"),
	('includeTop',"basic:Boolean",{'optional': True}),
	('weightsImagenet',"basic:Boolean",{'optional': True})]

	_output_ports = [("resnet50Model", "model", {'shape': 'diamond'})]

	def compute(self):
		from keras.applications.resnet50 import ResNet50
		from keras import optimizers
		from keras import losses

		
		includeTop = True
		wBool = False
		tBool = True

		weights = None	


		if 'includeTop' in self.inputPorts:
			tBool = self.get_input("includeTop")

		if 'weightsImagenet' in self.inputPorts:
			wBool = self.get_input("weightsImagenet")

		if(wBool):
			weights = 'imagenet'

		if tBool==False:
			includeTop = False


		metrics = self.get_input("metrics")
		
		optDict = {}

		optDict = self.get_input("optimizationDict")

		print(optDict['name'])

		opt = getattr(optimizers,optDict['name'])
		
		opt = opt()

		name = optDict['name']

		del(optDict['name'])

		for item in optDict:
				setattr(opt,item,optDict[item]) 
				print(item,getattr(opt,item))
		print("")

		x = self.get_input("firstNumberShape")
		y = self.get_input("secondNumberShape")
		gray = self.get_input("grayscale")
		c=3

		optDict['name'] = name

		if gray:
			c=1

		iShape = (x,y,c) 		
		nClass = self.get_input("numClass")


		model = ResNet50(weights=None,input_shape=iShape, classes=nClass)
		
		loss = self.get_input("loss")
		
		model.compile(loss=loss,optimizer=opt,metrics=[metrics])

		print(type(model))
		
		self.set_output("resnet50Model", model)



class inceptionV3(Module):
	_settings = ModuleSettings(namespace="Models")
	_input_ports = [IPort(name = "secondNumberShape", signature = "basic:Integer"),
	IPort(name = "firstNumberShape", signature = "basic:Integer"),
	IPort(name = "numClass", signature = "basic:Integer"),
	IPort(name = "metrics", signature = "basic:String"),
	IPort(name = "loss", signature = "basic:String"),
	IPort('optimizationDict',"basic:Dictionary"),
	IPort('grayscale',"basic:Boolean"),
	('includeTop',"basic:Boolean",{'optional': True}),
	('weightsImagenet',"basic:Boolean",{'optional': True})]

	_output_ports = [("inceptionV3Model", "model", {'shape': 'diamond'})]

	def compute(self):
		from keras.applications.inception_v3 import InceptionV3
		from keras import optimizers
		from keras import losses

		weights = None
		includeTop = True
		wBool = False
		tBool = True

	

		if 'includeTop' in self.inputPorts:
			tBool = self.get_input("includeTop")

		if 'weightsImagenet' in self.inputPorts:
			wBool = self.get_input("weightsImagenet")

		if(wBool):
			weights = 'imagenet'
		if(tBool==False):
			includeTop = False


		metrics = self.get_input("metrics")
		
		optDict = {}

		optDict = self.get_input("optimizationDict")

		print(optDict['name'])

		opt = getattr(optimizers,optDict['name'])
		
		opt = opt()

		name = optDict['name']

		del(optDict['name'])

		for item in optDict:
				setattr(opt,item,optDict[item]) 
				print(item,getattr(opt,item))
		print("")
		
		x = self.get_input("firstNumberShape")
		y = self.get_input("secondNumberShape")
		gray = self.get_input("grayscale")
		c=3
		
		optDict['name'] = name

		if gray:
			c=1

		iShape = (x,y,c) 		
		nClass = self.get_input("numClass")

		model = InceptionV3(weights=None,input_shape=iShape, classes=nClass)
		
		loss = self.get_input("loss")
		
		model.compile(loss=loss,optimizer=opt,metrics=[metrics])

		print(type(model))
		
		self.set_output("inceptionV3Model", model)

class VGG16(Module):
	_settings = ModuleSettings(namespace="Models")
	_input_ports = [IPort(name = "secondNumberShape", signature = "basic:Integer"),
	IPort(name = "firstNumberShape", signature = "basic:Integer"),
	IPort(name = "numClass", signature = "basic:Integer"),
	IPort(name = "metrics", signature = "basic:String"),
	IPort(name = "loss", signature = "basic:String"),
	IPort('optimizationDict',"basic:Dictionary"),
	IPort('grayscale',"basic:Boolean"),
	('includeTop',"basic:Boolean",{'optional': True}),
	('weightsImagenet',"basic:Boolean",{'optional': True})]

	_output_ports = [("VGG16Model", "model", {'shape': 'diamond'})]

	def compute(self):
		from keras.applications.vgg16 import VGG16
		from keras import optimizers
		from keras import losses

		weights = None
		includeTop = True
		wBool = False
		tBool = True


		if 'includeTop' in self.inputPorts:
			tBool = self.get_input("includeTop")

		if 'weightsImagenet' in self.inputPorts:
			wBool = self.get_input("weightsImagenet")

		if(wBool):
			weights = 'imagenet'
		if(tBool==False):
			includeTop = False


		metrics = self.get_input("metrics")
		
		optDict = {}

		optDict = self.get_input("optimizationDict")

		print(optDict['name'])

		opt = getattr(optimizers,optDict['name'])
		
		opt = opt()

		name = optDict['name']

		del(optDict['name'])

		for item in optDict:
				setattr(opt,item,optDict[item]) 
				print(item,getattr(opt,item))
		print("")
		
		x = self.get_input("firstNumberShape")
		y = self.get_input("secondNumberShape")
		gray = self.get_input("grayscale")
		c=3

		optDict['name'] = name
		
		if gray:
			c=1

		iShape = (x,y,c) 		
		nClass = self.get_input("numClass")

		model = VGG16(weights=None,input_shape=iShape, classes=nClass)
		
		loss = self.get_input("loss")
		
		model.compile(loss=loss,optimizer=opt,metrics=[metrics])

		print(type(model))
		
		self.set_output("VGG16Model", model)

class mobileNet(Module):
	_settings = ModuleSettings(namespace="Models")
	_input_ports = [IPort(name = "secondNumberShape", signature = "basic:Integer"),
	IPort(name = "firstNumberShape", signature = "basic:Integer"),
	IPort(name = "numClass", signature = "basic:Integer"),
	IPort(name = "metrics", signature = "basic:String"),
	IPort(name = "loss", signature = "basic:String"),
	IPort('optimizationDict',"basic:Dictionary"),
	IPort('grayscale',"basic:Boolean"),
	('includeTop',"basic:Boolean",{'optional': True}),
	('weightsImagenet',"basic:Boolean",{'optional': True})]

	_output_ports = [("mobileNetModel", "model", {'shape': 'diamond'})]

	def compute(self):
		from keras.applications.mobilenet import MobileNet
		from keras import optimizers
		from keras import losses


		weights = None
		includeTop = True
		wBool = False
		tBool = True


		if 'includeTop' in self.inputPorts:
			tBool = self.get_input("includeTop")

		if 'weightsImagenet' in self.inputPorts:
			wBool = self.get_input("weightsImagenet")

		if(wBool):
			weights = 'imagenet'
		if(tBool==False):
			includeTop = False


		metrics = self.get_input("metrics")
		
		optDict = {}

		optDict = self.get_input("optimizationDict")

		print(optDict['name'])

		opt = getattr(optimizers,optDict['name'])
		
		opt = opt()

		name = optDict['name']

		del(optDict['name'])

		for item in optDict:
				setattr(opt,item,optDict[item]) 
				print(item,getattr(opt,item))
		print("")
		
		x = self.get_input("firstNumberShape")
		y = self.get_input("secondNumberShape")
		gray = self.get_input("grayscale")
		c=3

		optDict['name'] = name
		
		if gray:
			c=1

		iShape = (x,y,c) 		
		nClass = self.get_input("numClass")

		model = MobileNet(weights=None,input_shape=iShape, classes=nClass)
		
		loss = self.get_input("loss")
		
		model.compile(loss=loss,optimizer=opt,metrics=[metrics])

		print(type(model))
		
		self.set_output("mobileNetModel", model)

class xception(Module):
	_settings = ModuleSettings(namespace="Models")
	_input_ports = [IPort(name = "secondNumberShape", signature = "basic:Integer"),
	IPort(name = "firstNumberShape", signature = "basic:Integer"),
	IPort(name = "numClass", signature = "basic:Integer"),
	IPort(name = "metrics", signature = "basic:String"),
	IPort(name = "loss", signature = "basic:String"),
	IPort('optimizationDict',"basic:Dictionary"),
	IPort('grayscale',"basic:Boolean"),
	('includeTop',"basic:Boolean",{'optional': True}),
	('weightsImagenet',"basic:Boolean",{'optional': True})]

	_output_ports = [("xceptionodel", "model", {'shape': 'diamond'})]

	def compute(self):
		from keras.applications.xception import Xception
		from keras import optimizers
		from keras import losses


		weights = None
		includeTop = True
		wBool = False
		tBool = True


		if 'includeTop' in self.inputPorts:
			tBool = self.get_input("includeTop")

		if 'weightsImagenet' in self.inputPorts:
			wBool = self.get_input("weightsImagenet")

		if(wBool):
			weights = 'imagenet'
		if(tBool==False):
			includeTop = False


		metrics = self.get_input("metrics")
		
		optDict = {}

		optDict = self.get_input("optimizationDict")

		print(optDict['name'])

		opt = getattr(optimizers,optDict['name'])
		
		opt = opt()

		name = optDict['name']

		del(optDict['name'])

		for item in optDict:
				setattr(opt,item,optDict[item]) 
				print(item,getattr(opt,item))
		print("")

		
		x = self.get_input("firstNumberShape")
		y = self.get_input("secondNumberShape")
		gray = self.get_input("grayscale")
		c=3

		optDict['name'] = name
		
		if gray:
			c=1

		iShape = (x,y,c) 		
		nClass = self.get_input("numClass")

		model = Xception(weights=None,input_shape=iShape, classes=nClass)
		
		loss = self.get_input("loss")
		
		model.compile(loss=loss,optimizer=opt,metrics=[metrics])

		print(type(model))
		
		self.set_output("mobileNetModel", model)

class VGG19(Module):
	_settings = ModuleSettings(namespace="Models")
	_input_ports = [IPort(name = "secondNumberShape", signature = "basic:Integer"),
	IPort(name = "firstNumberShape", signature = "basic:Integer"),
	IPort(name = "numClass", signature = "basic:Integer"),
	IPort(name = "metrics", signature = "basic:String"),
	IPort(name = "loss", signature = "basic:String"),
	IPort('optimizationDict',"basic:Dictionary"),
	IPort('grayscale',"basic:Boolean"),
	('includeTop',"basic:Boolean",{'optional': True}),
	('weightsImagenet',"basic:Boolean",{'optional': True})]

	_output_ports = [("VGG19Model", "model", {'shape': 'diamond'})]

	def compute(self):
		from keras.applications.vgg19 import VGG19
		from keras import optimizers
		from keras import losses


		weights = None
		includeTop = True
		wBool = False
		tBool = True


		if 'includeTop' in self.inputPorts:
			tBool = self.get_input("includeTop")

		if 'weightsImagenet' in self.inputPorts:
			wBool = self.get_input("weightsImagenet")

		if(wBool):
			weights = 'imagenet'
		if(tBool==False):
			includeTop = False


		metrics = self.get_input("metrics")
		
		optDict = {}

		optDict = self.get_input("optimizationDict")

		print(optDict['name'])

		opt = getattr(optimizers,optDict['name'])
		
		opt = opt()

		name = optDict['name']

		del(optDict['name'])

		for item in optDict:
				setattr(opt,item,optDict[item]) 
				print(item,getattr(opt,item))
		print("")
		
		x = self.get_input("firstNumberShape")
		y = self.get_input("secondNumberShape")
		gray = self.get_input("grayscale")
		c=3

		optDict['name'] = name
		
		if gray:
			c=1

		iShape = (x,y,c) 		
		nClass = self.get_input("numClass")

		model = VGG19(weights=None,input_shape=iShape, classes=nClass)
		
		loss = self.get_input("loss")
		
		model.compile(loss=loss,optimizer=opt,metrics=[metrics])

		print(type(model))
		
		self.set_output("mobileNetModel", model)

class inceptionResNetV2(Module):
	_settings = ModuleSettings(namespace="Models")
	_input_ports = [IPort(name = "secondNumberShape", signature = "basic:Integer"),
	IPort(name = "firstNumberShape", signature = "basic:Integer"),
	IPort(name = "numClass", signature = "basic:Integer"),
	IPort(name = "metrics", signature = "basic:String"),
	IPort(name = "loss", signature = "basic:String"),
	IPort('optimizationDict',"basic:Dictionary"),
	IPort('grayscale',"basic:Boolean"),
	('includeTop',"basic:Boolean",{'optional': True}),
	('weightsImagenet',"basic:Boolean",{'optional': True})]

	_output_ports = [("inceptionResNetV2Model", "model", {'shape': 'diamond'})]

	def compute(self):
		from keras.applications.inception_resnet_v2 import InceptionResNetV2
		from keras import optimizers
		from keras import losses


		weights = None
		includeTop = True
		wBool = False
		tBool = True


		if 'includeTop' in self.inputPorts:
			tBool = self.get_input("includeTop")

		if 'weightsImagenet' in self.inputPorts:
			wBool = self.get_input("weightsImagenet")

		if(wBool):
			weights = 'imagenet'
		if(tBool==False):
			includeTop = False


		metrics = self.get_input("metrics")
		
		optDict = {}

		optDict = self.get_input("optimizationDict")

		print(optDict['name'])

		opt = getattr(optimizers,optDict['name'])
		
		opt = opt()

		name = optDict['name']

		del(optDict['name'])

		for item in optDict:
				setattr(opt,item,optDict[item]) 
				print(item,getattr(opt,item))
		print("")
		
		x = self.get_input("firstNumberShape")
		y = self.get_input("secondNumberShape")
		gray = self.get_input("grayscale")
		c=3

		optDict['name'] = name
		
		if gray:
			c=1

		iShape = (x,y,c) 		
		nClass = self.get_input("numClass")

		model = InceptionResNetV2(weights=None,input_shape=iShape, classes=nClass)
		
		loss = self.get_input("loss")
		
		model.compile(loss=loss,optimizer=opt,metrics=[metrics])

		print(type(model))
		
		self.set_output("mobileNetModel", model)

class denseNet121(Module):
	_settings = ModuleSettings(namespace="Models")
	_input_ports = [IPort(name = "secondNumberShape", signature = "basic:Integer"),
	IPort(name = "firstNumberShape", signature = "basic:Integer"),
	IPort(name = "numClass", signature = "basic:Integer"),
	IPort(name = "metrics", signature = "basic:String"),
	IPort(name = "loss", signature = "basic:String"),
	IPort('optimizationDict',"basic:Dictionary"),
	IPort('grayscale',"basic:Boolean"),
	('includeTop',"basic:Boolean",{'optional': True}),
	('weightsImagenet',"basic:Boolean",{'optional': True})]

	_output_ports = [("denseNet121Model", "model", {'shape': 'diamond'})]

	def compute(self):
		from keras.applications.densenet import DenseNet121
		from keras import optimizers
		from keras import losses


		weights = None
		includeTop = True
		wBool = False
		tBool = True


		if 'includeTop' in self.inputPorts:
			tBool = self.get_input("includeTop")

		if 'weightsImagenet' in self.inputPorts:
			wBool = self.get_input("weightsImagenet")

		if(wBool):
			weights = 'imagenet'

		if(tBool==False):
			includeTop = False


		metrics = self.get_input("metrics")
		
		optDict = {}

		optDict = self.get_input("optimizationDict")

		print(optDict['name'])

		opt = getattr(optimizers,optDict['name'])
		
		opt = opt()

		name = optDict['name']

		del(optDict['name'])

		for item in optDict:
				setattr(opt,item,optDict[item]) 
				print(item,getattr(opt,item))
		print("")
		
		x = self.get_input("firstNumberShape")
		y = self.get_input("secondNumberShape")
		gray = self.get_input("grayscale")
		c=3

		optDict['name'] = name
		
		if gray:
			c=1

		iShape = (x,y,c) 		
		nClass = self.get_input("numClass")

		model = DenseNet121(weights=None,input_shape=iShape, classes=nClass)
		
		loss = self.get_input("loss")
		
		model.compile(loss=loss,optimizer=opt,metrics=[metrics])

		print(type(model))
		
		self.set_output("mobileNetModel", model)

class denseNet169(Module):
	_settings = ModuleSettings(namespace="Models")
	_input_ports = [IPort(name = "secondNumberShape", signature = "basic:Integer"),
	IPort(name = "firstNumberShape", signature = "basic:Integer"),
	IPort(name = "numClass", signature = "basic:Integer"),
	IPort(name = "metrics", signature = "basic:String"),
	IPort(name = "loss", signature = "basic:String"),
	IPort('optimizationDict',"basic:Dictionary"),
	IPort('grayscale',"basic:Boolean"),
	('includeTop',"basic:Boolean",{'optional': True}),
	('weightsImagenet',"basic:Boolean",{'optional': True})]

	_output_ports = [("denseNet169Model", "model", {'shape': 'diamond'})]

	def compute(self):
		from keras.applications.densenet import MobileNet
		from keras import optimizers
		from keras import losses

		weights = None
		includeTop = True
		wBool = False
		tBool = True


		if 'includeTop' in self.inputPorts:
			tBool = self.get_input("includeTop")

		if 'weightsImagenet' in self.inputPorts:
			wBool = self.get_input("weightsImagenet")

		if(wBool):
			weights = 'imagenet'
		if(tBool==False):
			includeTop = False

		metrics = self.get_input("metrics")
		
		optDict = {}

		optDict = self.get_input("optimizationDict")

		print(optDict['name'])

		opt = getattr(optimizers,optDict['name'])
		
		opt = opt()

		name = optDict['name']

		del(optDict['name'])

		for item in optDict:
				setattr(opt,item,optDict[item]) 
				print(item,getattr(opt,item))
		print("")
		
		x = self.get_input("firstNumberShape")
		y = self.get_input("secondNumberShape")
		gray = self.get_input("grayscale")
		c=3

		optDict['name'] = name
		
		if gray:
			c=1

		iShape = (x,y,c) 		
		nClass = self.get_input("numClass")


		model = DenseNet169(weights=None,input_shape=iShape, classes=nClass)
		
		loss = self.get_input("loss")
		
		model.compile(loss=loss,optimizer=opt,metrics=[metrics])

		print(type(model))
		
		self.set_output("mobileNetModel", model)

class denseNet201(Module):
	_settings = ModuleSettings(namespace="Models")
	_input_ports = [IPort(name = "secondNumberShape", signature = "basic:Integer"),
	IPort(name = "firstNumberShape", signature = "basic:Integer"),
	IPort(name = "numClass", signature = "basic:Integer"),
	IPort(name = "metrics", signature = "basic:String"),
	IPort(name = "loss", signature = "basic:String"),
	IPort('optimizationDict',"basic:Dictionary"),
	('includeTop',"basic:Boolean",{'optional': True}),
	('weightsImagenet',"basic:Boolean",{'optional': True}),	
	IPort('grayscale',"basic:Boolean")]

	_output_ports = [("denseNet201Model", "model", {'shape': 'diamond'})]

	def compute(self):
		from keras.applications.densenet import MobileNet
		from keras import optimizers
		from keras import losses

		weights = None
		includeTop = True
		wBool = False
		tBool = True


		if 'includeTop' in self.inputPorts:
			tBool = self.get_input("includeTop")

		if 'weightsImagenet' in self.inputPorts:
			wBool = self.get_input("weightsImagenet")

		if(wBool):
			weights = 'imagenet'
		if(tBool==False):
			includeTop = False

		metrics = self.get_input("metrics")
		
		optDict = {}

		optDict = self.get_input("optimizationDict")

		print(optDict['name'])

		opt = getattr(optimizers,optDict['name'])
		
		opt = opt()

		name = optDict['name']

		del(optDict['name'])

		for item in optDict:
				setattr(opt,item,optDict[item]) 
				print(item,getattr(opt,item))
		print("")
		
		x = self.get_input("firstNumberShape")
		y = self.get_input("secondNumberShape")
		gray = self.get_input("grayscale")
		c=3

		optDict['name'] = name
		
		if gray:
			c=1

		iShape = (x,y,c) 		
		nClass = self.get_input("numClass")


		model = DenseNet201(weights=None,input_shape=iShape, classes=nClass)
		
		loss = self.get_input("loss")
		
		model.compile(loss=loss,optimizer=opt,metrics=[metrics])

		print(type(model))
		
		self.set_output("mobileNetModel", model)
   

       
        



class modelFit(Module):
	_settings = ModuleSettings(namespace="Models")
	_input_ports = [IPort(name = "trainDataset", signature = "ImageDataGenerator"),
	IPort(name = "testDataset", signature = "ImageDataGenerator"),
	("genericModel","model",{'shape': 'Diamond'}),
	IPort(name = "stepsEpoch", signature = "basic:Integer"),
	IPort(name = "numEpochs", signature = "basic:Integer"),
	IPort(name = "valSteps", signature = "basic:Integer"),]

	def compute(self):

		train_generator = self.get_input("trainDataset")
		test_generator = self.get_input("testDataset")

		stepsEpoch = self.get_input("stepsEpoch")
		numEpochs = self.get_input("numEpochs")
		valSteps = self.get_input("valSteps")

		model = self.get_input("genericModel")


		print("")
		print("loss:")
		print(model.loss)
		print("")
		met = model.metrics[0]
		met = met.capitalize()
		print("metrics:")
		print(met)
		print("")


		history = model.fit_generator(
        train_generator,
        steps_per_epoch=stepsEpoch,
        epochs=numEpochs,
        validation_data=test_generator,        
        validation_steps=valSteps)


		plt.subplot(221)
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('Accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		
		

		plt.subplot(222)
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('Log-Loss (Cost Function)')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')

		mng = plt.get_current_fig_manager()
		mng.window.showMaximized()

		plt.show()
		


        
	

_modules = [ImageDataGenerator,augmentationConf,preprocessingConf,imageVisualization,resnet50,sgdOptimization,rmsPropOptimization,
model,modelFit,nadamOptimization,adagradOptimization,adadeltaOptimization,adamOptimization,adamaxOptimization,imgLoader,
mobileNet,inceptionV3,VGG16,xception,VGG19,inceptionResNetV2,denseNet121,denseNet169,denseNet201]


