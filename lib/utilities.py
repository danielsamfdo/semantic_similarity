import pickle
import dill
import lib.preprocess as process
from collections import Counter
import numpy as np

def dict_dotprod(d1, d2):
  """Return the dot product (aka inner product) of two vectors, where each is
  represented as a dictionary of {index: weight} pairs, where indexes are any
  keys, potentially strings.  If a key does not exist in a dictionary, its
  value is assumed to be zero."""
  smaller = d1 if len(d1)<len(d2) else d2  # BUGFIXED 20151012
  total = 0
  for key in smaller.iterkeys():
      total += d1.get(key,0) * d2.get(key,0)
  return total

def save_weights(path,weights):
  prefix ="weights/"
  pickle.dump(weights, open(prefix+path, 'wb'))

def load_weights(path):
  return pickle.load(open(path, 'rb'))

def evaluate(gold_standard, predicted_answers):
  return np.sum(np.absolute(np.array(gold_standard) - np.array(predicted_answers)))/len(gold_standard) 
  # return math.sqrt(np.sum(np.power(np.array(gold_standard) - np.array(predicted_answers),2))) 

def get_dict_vector_of_2_sentences(sentence_1_tokens, sentence_2_tokens):
  return dict(Counter(sentence_1_tokens)+Counter(sentence_1_tokens))

def get_dict_vectors_of_documents(documents):
  init_doc_count = len(documents)/2
  operated_doc_count = 0
  doc_dict_vectors_list = []
  for i in range(len(documents)/2):
    if(operated_doc_count == 400):
      init_doc_count-=400
      operated_doc_count=0
      print str(init_doc_count) + " sets remaining"
    operated_doc_count+=1
    document1, document2 = documents[(2*i)], documents[(2*i)+1]
    sent_1_tokens = process.tokens(document1)
    sent_2_tokens = process.tokens(document2)
    # print sent_1_tokens, sent_2_tokens
    doc_dict_vectors_list.append(get_dict_vector_of_2_sentences(sent_1_tokens, sent_2_tokens))
  return doc_dict_vectors_list