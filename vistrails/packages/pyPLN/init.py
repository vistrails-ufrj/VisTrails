from vistrails.core.modules.vistrails_module import Module, ModuleError
from vistrails.core.system import list2cmdline

import vistrails.core.requirements
import os


def package_requirements():
	if vistrails.core.requirements.python_module_exists('nltk'):
		print ("Nltk is already installed and ready to use")
	else:
		raise vistrails.core.requirements.MissingRequirement('nltk')

class NltkCorpus(Module):

	def compute(self):
		cmd = ['sudo','python','-m',' nltk.downloader']
		cmdline = list2cmdline(cmd)
		print("You're about to download all the corpus included in NLTK packages. This may take some minutes")
		print '_DEBUG' + str(cmdline)

		if (os.system(cmdline)):
			print "Done"
		else:
			raise ModuleError(self, "Execution Failed")

class PrintTest(Module):
	
	def compute(self):
		from nltk.corpus import brown
		test = brown.words()
		print (test[:10])
		print "This is our Test Module"

# class OwnCorpus(Module):

# 	def compute(self):
# 		from nltk.corpus import PlaintextCorpusReader
# 		corpus_path = '' # Path
# 		wordlists = PlaintextCorpusReader(corpus_path, '.*')

# 		# 

# 		# 


_modules = [NltkCorpus, PrintTest,]
