from vistrails.core.modules.vistrails_module import Module, ModuleError
from vistrails.core.system import list2cmdline
from vistrails.core.modules.config import IPort, OPort

import os

# Module to type the Corpus
# Parameters:
# Return
class Corpus(Module):
	corpus_dir = None
	corpus = None
	def __init__(self, corpus_dir):
		self.corpus_dir = corpus_dir
	
		self.corpus = self.load_corpus_dir()
		# print (check_corpus())

	def load_corpus_dir(self):
			from nltk.corpus import BracketParseCorpusReader

			corpus_root = str(self.corpus_dir.name)
			file_patt =  r".*/.*\.txt"
			corpus =  BracketParseCorpusReader(corpus_root, file_patt) 

			# print ("These are the corpus' Files")
			# files_ids = corpus_list.fileids()
			# for id in (corpus.fileids()): # corpus.words('connectives')
			# 	print id
			
			return corpus

	def check_corpus(self):
		files_ids = self.corpus.fileids()

		if len(files_ids) > 0:
			print (files_ids[:10])
			return 1

		return 0

# Module set up the content of ntlk to user's acess
# Parameters: None
# Return : None
class UpdateNltkCorpus(Module):

	def compute(self):
		cmd = ['sudo','python','-m',' nltk.downloader']
		cmdline = list2cmdline(cmd)
		print("You're about to download all the corpus included in NLTK packages. This may take some minutes")
		print '_DEBUG' + str(cmdline)

		if (os.system(cmdline)):
			print "Done"
		else:
			raise ModuleError(self, "Execution Failed")

# Module:
# Parameters:
# Return: 
class ShowNLTKCorpus(Module):
	_output_ports = [OPort(name = "output_corpus", signature = "basic:List")]
	def compute(self):

		abs_dir = os.path.dirname(os.path.realpath(__file__))
		rel_path = "corpuses.txt"
		abs_file_path = os.path.join(abs_dir, rel_path)
		
		corpuses_file = open(abs_file_path,'r')
		
		corpuses_list = []
		for corpus in corpuses_file:
			corpuses_list.append(corpus)
		print "There are this corpuses available from Nltk Package"
		print " " + str(corpuses_list) + " "
		self.set_output('output_corpus', corpuses_list)

# Module
# Parameters:
# Return
class LoadCorpus(Module):
		
		_input_ports = [IPort(name = "input_corpus", signature = "basic:Directory")]
		_output_ports = [OPort(name = "output_corpus", signature = "Corpus")]
		def compute(self):
			
			corpus_dir = self.get_input('input_corpus')
			corpus = Corpus(corpus_dir)
			# corpus.check_corpus()
			self.set_output('output_corpus', corpus)

# Module
# Parameters:
# Return


# class OwnCorpus(Module):

# 	def compute(self):
# 		from nltk.corpus import PlaintextCorpusReader
# 		corpus_path = '' # Path
# 		wordlists = PlaintextCorpusReader(corpus_path, '.*')

# 		# 

# 		# 


_modules = [UpdateNltkCorpus, ShowNLTKCorpus,LoadCorpus, Corpus]
