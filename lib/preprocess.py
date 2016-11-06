from nltk.stem import WordNetLemmatizer
from functools32 import lru_cache

def tokens(sentence):
	sentence.split()

def lemmatize(tokens)
	wnl = WordNetLemmatizer()
	return [wnl.lemmatize(token) for token in tokens]

# http://stackoverflow.com/questions/16181419/is-it-possible-to-speed-up-wordnet-lemmatizer
# wnl = WordNetLemmatizer()
# lemmatize = lru_cache(maxsize=50000)(wnl.lemmatize)
