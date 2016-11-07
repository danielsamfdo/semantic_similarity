from collections import defaultdict

def ngram_vector_keys(tokens, ngram_size=1):
  vector_keys = []
  for i in range(len(tokens)-(ngram_size-1)):
    vector_keys.append(" ".join(tokens[i:i+(ngram_size)]))
  return vector_keys

def JaccardCoefficient(sent_1_tokens, sent_2_tokens, ngram_size=1):
  set1 = set(ngram_vector_keys(sent_1_tokens, ngram_size))
  set2 = set(ngram_vector_keys(sent_2_tokens, ngram_size))
  return float(len(set1.intersection(set2)))/len(set1.union(set2))
