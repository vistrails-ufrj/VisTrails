import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag

raw = "Hi, I'm John."
raw = raw.strip()
print "Raw Text" + str(raw)

sent = sent_tokenize(raw)
print sent
