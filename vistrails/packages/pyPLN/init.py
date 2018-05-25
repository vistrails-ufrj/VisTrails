from vistrails.core.modules.vistrails_module import Module, ModuleError
from vistrails.core.system import list2cmdline
from vistrails.core.modules.config import IPort, OPort

import os

############## Comment Pattern ##############

# Module:
# Parameters:
# Return:

	# Method:
	# Parameters:
	# Return:

##############

############## Workflow Types ###############

# Module to create Corpus as parameterer's type
# Parameters: (Directory) A directory to load the corpus from 
# Return:
class Corpus(Module):
	corpus_dir = None
	corpus = None
	def __init__(self, corpus_obj = None, corpus_dir = None):
		self.corpus_dir = corpus_dir
		
		if ( corpus_dir != None):
			self.corpus = self.load_corpus_dir()

		if corpus_obj != None:
			self.corpus = corpus_obj
		# print (check_corpus())

	# Method to load the corpus 
	# Parameters:
	# Return: The Nltk.Corpus structure 
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

	# Method to load the corpus 
	# Parameters:
	# Return: The Nltk.Corpus structure 
	def get_raw(self, fileid):
		return corpus.raw(fileid)

	def get_textids(self):
		return corpus.fileids()
	# Method to check the corpus (debug)
	# Parameters:
	# Return: 1 for existing corpus, 0 for not
	def check_corpus(self):
		files_ids = self.corpus.fileids()

		if len(files_ids) > 0:
			print (files_ids[:10])
			return 1

		return 0




# Module to create Tokens as parameters' type
# Parameters: (String) A raw text to tokenize 
# Return:
class Tokens(Module):

	def __init__(self, raw):
		self.raw = raw
		self.tok_sen = self.tokenize(raw)
		self._stemmed = False
		self._normalized = False
		self.tok_tag = None

	# Method to tokenize a text
	# Parameters: (String) Text
	# Return:
	def tokenize(self, raw):
		from nltk import word_tokenize

		self.tok_sen = word_tokenize(raw)

		return self.tok_sen

	def set_tokens(self, tokens):
		self.tok_sen = tokens

	def get_tokens(self, tokens):
		return self.tok_sen

	def get_tok_tag(self):
		return self.tok_tag


	# Method to Stem by Porter Algorithm a set of tokens
	# Parameters:
	# Return:
	def PorterStem(self):
		import nltk

		if not self._stemmed:
			porter = nltk.PorterStemmer()
			for i in range(len(self.tok_sen)):
				self.tok_sen[i] = porter.stem(self.tok_sen[i])

			self._stemmed = True
		else:
			print "Already Stemmed"


	# Method to Stem by Lancaster Algorithm a set of tokens
	# Parameters:
	# Return:
	def LancasterStem(self):
		import nltk

		print "test"
		if not self._stemmed:
			lancaster = nltk.LancasterStemmer()
			print 'test'
			for i in range(len(self.tok_sen)):
				self.tok_sen[i] = lancaster.stem(self.tok_sen[i])
			print 'test'
			print self.tok_sen
			self._stemmed = True
		else:
			print "Already Stemmed"


	# Method to Lemmatize a set of tokens
	# Parameters:
	# Return:
	def WordNetLemma(self):
		import nltk

		if not self._stemmed:
			wn = nltk.WordNetLemmatizer()
			for i in range(len(self.tok_sen)):
				self.tok_sen[i] = wn.lemmatize(self.tok_sen[i])

			self._stemmed = True
		else:
			print "Already Stemmed"

	# Method to normalize the set of tokens
	# Parameters: 
	# Return:		
	def normalize(self):
		norm_tokens = [token.lower() for token in self.tok_sen]
		self.tok_sen = norm_tokens
		self._normalized = True

	def rmv_stopwords(self, stopwords):
		if not self._normalized:
			self.normalize()

		norm_tokens = [tokens for tokens in self.tok_sen if tokens not in stopwords]
		self.tok_sen = norm_tokens

	def postag(self):
		import nltk

		self.tok_tag = nltk.pos_tag(self.tok_sen)

	# Method to tag Part of Speech tokens
	# Parameters: 
	# Return:
	def stanfordTag(self,jar_path,model_path):
		from nltk.tag import StanfordPOSTagger
		spt = StanfordPOSTagger(model_path,jar_path)
		self.tok_tag = spt.tag(self.tok_sen)



##############

############## Workflow Modules ##############

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

# Module presents the NLTK's provided Corpuses
# Parameters:
# Return: 
class ShowNLTKCorpus(Module):
	_output_ports = [OPort(name = "output_corpus", signature = "basic:List")]
	def compute(self):

		abs_dir = os.path.dirname(os.path.realpath(__file__))
		rel_path = "corpuses.txt"
		abs_file_path = os.path.join(abs_dir, rel_path)
		
		corpuses_file = open(abs_file_path,'r')
		
		corpuses_list = [corpus.strip('\n') for corpus in corpuses_file]

		print "There are this corpuses available from Nltk Package"
		print " " + str(corpuses_list) + " "
		self.set_output('output_corpus', corpuses_list)

# Module loads a corpus object by a specified corpus in NLTK
# Parameters: (string) corpus name
# Return: (Corpus) corpus object
class LoadNLTKCorpus(Module):
	_input_ports = [IPort(name = "input_corpus", signature = "basic:String")]
	_output_ports = [OPort(name = "output_corpus_object", signature = "Corpus")]

	def compute(self):
		corpus_str = self.get_input('input_corpus')

		corpus = self.get_corpus(corpus_str)

		self.set_output('output_corpus_object',corpus)

	def get_corpus(self, corpus_str):
		import nltk.corpus as ncp

		corpus = getattr(ncp,corpus_str)
		corpus_obj = Corpus(corpus_obj = corpus)

		return corpus_obj 

# Module load the user's provided corpus
# Parameters:
# Return
class LoadMyCorpus(Module):
		
		_input_ports = [IPort(name = "input_corpus", signature = "basic:Directory")]
		_output_ports = [OPort(name = "output_corpus", signature = "Corpus")]
		def compute(self):
			
			corpus_dir = self.get_input('input_corpus')
			corpus = Corpus(corpus_dir = corpus_dir)
			# corpus.check_corpus()
			self.set_output('output_corpus', corpus)


class Tokenizer(Module):

	_input_ports = [IPort("input_text", "basic:String")]
	_output_ports = [OPort("output_tokens", "Tokens")]

	def compute(self):
		raw = self.get_input("input_text")
		# raw = corpusObject.corpus.raw()
		tokObject = Tokens(raw)

		self.set_output('output_tokens', tokObject)

class PorterStemmer(Module):
	_input_ports = [IPort(name = "input_tokens", signature = "Tokens")]
	_output_ports = [OPort(name = "output_token_stemmed", signature = "Tokens")]

	def compute(self):
		tokens = self.get_input("input_tokens")

		tokens.PorterStem()

		self.set_output("output_token_stemmed",tokens)


class LancasterStemmer(Module):
	_input_ports = [IPort("input_tokens", "Tokens")]
	_output_ports = [OPort("output_token_stemmed", "Tokens")]

	def compute(self):
		tokens = self.get_input("input_tokens")

		tokens.LancasterStem()

		self.set_output("output_token_stemmed",tokens)


class WordNetLemmatizer(Module):
	_input_ports = [IPort("input_tokens", "Tokens")]
	_output_ports = [OPort("output_token_lemma", "Tokens")]

	def compute(self):
		tokens = self.get_input("input_tokens")

		tokens.WordNetLemma()

		self.set_output("output_token_lemma",tokens)

class Normalize(Module):
	_input_ports = [IPort("input_tokens", "Tokens")]
	_output_ports = [OPort("output_token_norm", "Tokens")]

	def compute(self):
		tokens = self.get_input("input_tokens")

		tokens.normalize()

		self.set_output("output_token_norm", tokens)

class Rmv_Stopwords(Module):
	_input_ports = [IPort("input_tokens", "Tokens"), IPort(name = 'input_stopwords',signature = 'basic:List', optional = True)]
	_output_ports = [OPort("output_token_norm", "Tokens")]

	def compute(self):
		tokens = self.get_input("input_tokens")
		stopwords = None
		if self.has_input('input_stopwords'):
			stopwords = self.get_input('input_stopwords')
		else:
			stopwords = self.get_nltk_stopw()

		tokens.rmv_stopwords(stopwords)

		self.set_output("output_token_norm", tokens)

	def get_nltk_stopw(self, lang="english"):
		import nltk

		return nltk.corpus.stopwords.words(lang)

class defaultPOStagger(Module):
	_input_ports = [IPort('input_tokens', "Tokens")]
	_output_ports = [OPort('output_token', 'Tokens')]

	def compute(self):
		tokens = self.get_input('input_tokens')
		tokens.postag()

		print tokens.get_tok_tag()
		self.set_output('output_token', tokens)


class StanfordPOStagger(Module):
	_input_ports = [IPort('path_to_jar','basic:Path'), 
					IPort('path_to_model','basic:Path'),
					IPort('input_tokens', "Tokens")]
	_output_ports = [OPort('output_token', 'Tokens')]

	def compute(self):
		tokens = self.get_input('input_tokens')
		jar_path = str(self.get_input('path_to_jar').name)
		model_path = str(self.get_input('path_to_model').name)

		tokens.stanfordTag(jar_path,model_path)

		print tokens.get_tok_tag()
		self.set_output('output_token', tokens)


##############




_modules = [UpdateNltkCorpus, ShowNLTKCorpus, LoadMyCorpus, LoadNLTKCorpus, Corpus, 
			Tokens, Tokenizer, 
			PorterStemmer, LancasterStemmer, WordNetLemmatizer,
			Normalize, Rmv_Stopwords,
			defaultPOStagger, StanfordPOStagger
			]
